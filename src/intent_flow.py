from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from .cache import FileCache
from .clustering import cluster_embeddings, split_large_clusters
from .embeddings import embed_texts
from .openai_client import chat_json_with_cache


HIGH_IMPACT_INTENTS = ["billing", "compliance", "renewal", "account"]

SYSTEM_PROMPT = "You standardize intents and create bot flow templates. Always respond in English."
USER_TEMPLATE = (
    "Given multiple flow_candidates (intent_candidate, scenario_candidate, flow_steps_candidate, resolution),\n"
    "produce ONLY JSON:\n"
    "{{\n"
    "  'intent_name': string,\n"
    "  'definition': string,\n"
    "  'scenarios': [\n"
    "    {{\n"
    "      'scenario_name': string,\n"
    "      'frequency': number,\n"
    "      'flow_template': [\n"
    "        {{ 'type': 'ask|inform|action|handover', 'text': string }}\n"
    "      ],\n"
    "      'required_fields': string[],\n"
    "      'handover_rules': string[],\n"
    "      'example_conversation_ids': string[]\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
    "Rules:\n"
    "- Use semantic, reusable steps (bot-friendly)\n"
    "- Do NOT include UI click paths unless unavoidable\n"
    "- required_fields: inputs needed to complete flow (e.g., account_id, invoice_no)\n"
    "- handover_rules: when to escalate to human\n\n"
    "Flow candidates:\n{candidates}\n"
)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        cleaned = (item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _cluster_flow_keys(
    flow_items: List[Dict[str, Any]],
    client: OpenAI | None,
    embed_model: str,
    embed_cache: FileCache,
    threshold: float,
) -> List[List[int]]:
    keys = [item["flow_key"] for item in flow_items]
    embeddings, _ = embed_texts(keys, client, embed_model, embed_cache)
    return split_large_clusters(cluster_embeddings(embeddings, threshold), embeddings)


def _has_high_impact(intent_text: str) -> bool:
    lowered = intent_text.lower()
    return any(term in lowered for term in HIGH_IMPACT_INTENTS)


def _fallback_intent(cluster_items: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    first = cluster_items[0]
    intent_name = first.get("intent_candidate") or "General Inquiry"
    scenario_name = first.get("scenario_candidate") or "General"
    flow_steps = first.get("flow_steps_candidate") or []
    flow_template = [{"type": "ask", "text": step} for step in flow_steps] if flow_steps else []
    convo_ids = _dedupe([item.get("conversation_id", "") for item in cluster_items])
    return {
        "intent_id": f"intent_{idx}",
        "intent_name": intent_name,
        "definition": f"Handling requests related to {intent_name.lower()}.",
        "confidence": {
            "score_0_to_1": 0.3,
            "reason": "fallback_without_llm",
        },
        "automation_suitability": {
            "suitable": False,
            "score_0_to_1": 0.2,
            "reason": "fallback_without_llm",
        },
        "scenarios": [
            {
                "scenario_name": scenario_name,
                "frequency": len(cluster_items),
                "flow_template": flow_template,
                "required_inputs": [],
                "handover_rules": [],
                "example_conversation_ids": convo_ids[:5],
            }
        ],
    }


def build_intent_flow_catalogue(
    atoms_rows: List[Dict[str, Any]],
    client: OpenAI | None,
    llm_model: str,
    llm_cache: FileCache,
    embed_model: str,
    embed_cache: FileCache,
    system_prompt: str = SYSTEM_PROMPT,
    user_template: str = USER_TEMPLATE,
    threshold: float = 0.82,
    min_freq: int = 3,
) -> List[Dict[str, Any]]:
    flow_items: List[Dict[str, Any]] = []
    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id")
        for candidate in atoms.get("flow_candidates", []) or []:
            intent = (candidate.get("intent_candidate") or "").strip()
            scenario = (candidate.get("scenario_candidate") or "").strip()
            if not intent and not scenario:
                continue
            flow_items.append(
                {
                    "conversation_id": convo_id,
                    "intent_candidate": intent,
                    "scenario_candidate": scenario,
                    "flow_steps_candidate": candidate.get("flow_steps_candidate") or [],
                    "resolution": candidate.get("resolution") or {},
                    "flow_key": f"{intent} | {scenario}".strip(" |"),
                }
            )

    if not flow_items:
        return []

    clusters = _cluster_flow_keys(flow_items, client, embed_model, embed_cache, threshold)
    results: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        cluster_items = [flow_items[i] for i in cluster]
        freq = len(cluster_items)
        intent_text = " ".join(item["intent_candidate"] for item in cluster_items)
        should_llm = client is not None and (freq >= min_freq or _has_high_impact(intent_text))

        if should_llm:
            payload = {
                "cluster_id": f"cluster_{idx}",
                "cluster_size": freq,
                "flow_candidates": [
                    {
                        "intent_candidate": item.get("intent_candidate", ""),
                        "scenario_candidate": item.get("scenario_candidate", ""),
                        "flow_steps_candidate": item.get("flow_steps_candidate", []),
                        "resolution": item.get("resolution", {}),
                    }
                    for item in cluster_items
                ],
            }
            candidates_json = json.dumps(payload, ensure_ascii=True)
            if "{candidates}" in user_template:
                prompt = user_template.replace("{candidates}", candidates_json)
            else:
                prompt = f"{user_template}\n\nFlow candidates:\n{candidates_json}"
            key = llm_cache.key_for(llm_model, system_prompt, prompt)
            try:
                parsed = chat_json_with_cache(client, llm_model, system_prompt, prompt, llm_cache, key)
                scenarios = parsed.get("scenarios") or []
                convo_ids = _dedupe([item.get("conversation_id", "") for item in cluster_items])
                for scenario in scenarios:
                    scenario.setdefault("frequency", freq)
                    scenario.setdefault("required_inputs", [])
                    scenario.setdefault("handover_rules", [])
                    scenario.setdefault("example_conversation_ids", convo_ids[:5])
                parsed.setdefault("intent_id", f"intent_{idx}")
                parsed.setdefault(
                    "confidence",
                    {"score_0_to_1": 0.0, "reason": "missing_confidence_from_prompt"},
                )
                parsed.setdefault(
                    "automation_suitability",
                    {"suitable": False, "score_0_to_1": 0.0, "reason": "missing_suitability_from_prompt"},
                )
                parsed["scenarios"] = scenarios
                results.append(parsed)
                continue
            except Exception:  # noqa: BLE001
                pass

        results.append(_fallback_intent(cluster_items, idx))

    return results
