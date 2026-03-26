from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=float)
    response = client.embeddings.create(model=model, input=texts)
    vectors = [row.embedding for row in response.data]
    max_len = max(len(v) for v in vectors) if vectors else 0
    cleaned = []
    for vec in vectors:
        if len(vec) < max_len:
            vec = vec + [0.0] * (max_len - len(vec))
        elif len(vec) > max_len:
            vec = vec[:max_len]
        cleaned.append(vec)
    return np.array(cleaned, dtype=float)


def build_faq_index(
    faq_payload: Any, client: OpenAI, model: str
) -> Dict[str, Any]:
    faqs = faq_payload.get("faqs", []) if isinstance(faq_payload, dict) else faq_payload or []
    items = []
    texts = []
    for faq in faqs:
        if not isinstance(faq, dict):
            continue
        canonical = (faq.get("canonical_question") or "").strip()
        variants = faq.get("question_variants") or []
        joined = " ".join([canonical] + [str(v) for v in variants if v])
        if not joined.strip():
            continue
        items.append(
            {
                "faq_id": faq.get("faq_id"),
                "canonical_question": canonical,
                "answer": faq.get("answer") or faq.get("draft_answer") or "",
                "confidence": faq.get("confidence"),
                "needs_verification": faq.get("needs_verification"),
                "tags": faq.get("tags") or [],
            }
        )
        texts.append(joined)
    embeddings = _embed_texts(client, model, texts)
    return {
        "items": items,
        "embeddings": embeddings,
        "embeddings_norm": _normalize_matrix(embeddings),
    }


def build_scenario_index(
    intent_payload: Any, client: OpenAI, model: str
) -> Dict[str, Any]:
    intents = intent_payload.get("intents", []) if isinstance(intent_payload, dict) else intent_payload or []
    items = []
    texts = []
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        intent_name = (intent.get("intent_name") or "").strip()
        definition = (intent.get("definition") or "").strip()
        scenarios = intent.get("scenarios", []) or []
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            scenario_name = (scenario.get("scenario_name") or "").strip()
            required_fields = scenario.get("required_fields") or scenario.get("required_inputs") or []
            handover_rules = scenario.get("handover_rules") or []
            example_asks = scenario.get("example_asks") or []
            flow_steps = scenario.get("flow_template") or []
            step_texts = []
            for step in flow_steps:
                if isinstance(step, dict):
                    step_texts.append(str(step.get("text") or ""))
                else:
                    step_texts.append(str(step))
            scenario_text = " ".join(
                [
                    intent_name,
                    definition,
                    scenario_name or "Scenario",
                    " ".join([str(f) for f in required_fields if f]),
                    " ".join([str(r) for r in handover_rules if r]),
                    " ".join([str(a) for a in example_asks if a]),
                    " ".join([t for t in step_texts if t]),
                ]
            ).strip()
            if not scenario_text:
                continue
            items.append(
                {
                    "intent_name": intent_name,
                    "intent_definition": definition,
                    "scenario_name": scenario_name,
                    "required_fields": required_fields,
                    "handover_rules": handover_rules,
                    "flow_template": flow_steps,
                    "example_asks": example_asks,
                }
            )
            texts.append(scenario_text)
    embeddings = _embed_texts(client, model, texts)
    return {
        "items": items,
        "embeddings": embeddings,
        "embeddings_norm": _normalize_matrix(embeddings),
    }


def search_index(
    query: str, index: Dict[str, Any], client: OpenAI, model: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    if not query or not index or not index.get("items"):
        return []
    embeddings_norm = index.get("embeddings_norm")
    if embeddings_norm is None or embeddings_norm.size == 0:
        return []
    query_vec = _embed_texts(client, model, [query])
    if query_vec.size == 0:
        return []
    query_norm = _normalize_matrix(query_vec)
    scores = embeddings_norm @ query_norm[0]
    if scores.size == 0:
        return []
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        item = index["items"][int(idx)]
        results.append({"item": item, "score": float(scores[int(idx)])})
    return results
