from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .cache import FileCache
from .models import Atoms
from .openai_client import chat_json_with_cache


SYSTEM_PROMPT = (
    "You extract FAQ candidates, flow candidates, and blueprint signals from customer-agent conversations. "
    "Be faithful. Do not invent facts or pricing. Always respond in English."
)
ATOMS_SCHEMA = """Atom schema:
{{
  "conversation_id": "string",
  "faq_candidates": [
    {{
      "q_raw": "string",
      "q_clean": "string",
      "a_candidates": ["string"],
      "evidence": {{ "q_turn": [int,int], "a_turn": [int,int] }}
    }}
  ],
  "flow_candidates": [
    {{
      "intent_candidate": "string",
      "scenario_candidate": "string",
      "flow_steps_candidate": ["string"],
      "resolution": {{ "status": "resolved|pending|unresolved|unknown", "summary": "string" }},
      "evidence": {{ "turn_span": [int,int] }}
    }}
  ],
  "blueprint_signals": {{
    "policy": ["string"],
    "tone": ["string"],
    "escalations": ["string"]
  }}
}}
Return ONLY valid JSON matching this schema. Do not add extra fields."""
USER_TEMPLATE = (
    "Input is a normalized conversation JSON with messages_for_llm[] in order.\n"
    "Return ONLY valid JSON matching the Atom Schema.\n\n"
    "STRICT RULES\n"
    "1) FAQ candidate quality gate:\n"
    "- Include ONLY real customer asks that are self-contained product/service questions or requests.\n"
    "- EXCLUDE fillers/navigation/repair prompts unless they include a domain noun (ivr/call flow/billing/etc).\n"
    "- Rewrite q_clean to be self-contained by adding missing nouns from nearby context.\n"
    "- Max 3 faq_candidates per conversation. Pick the most important.\n\n"
    "2) Answer candidates:\n"
    "- For each faq_candidate, output 1–3 a_candidates IF agent provided guidance/information.\n"
    "- If agent did NOT provide a real answer, output a_candidates = [\"NEEDS_INFO: <exact missing info required>\"].\n"
    "- a_candidates must be grounded in agent messages. No outside facts.\n\n"
    "3) Evidence anchoring:\n"
    "- q_turn and a_turn are inclusive 0-based ranges of indices in messages_for_llm[].\n"
    "- q_turn must cover the customer question turns.\n"
    "- a_turn must cover the agent answer turns.\n"
    "- Keep spans tight.\n\n"
    "4) Flow candidates:\n"
    "- Create 1–2 flow_candidates (max 2) per conversation per distinct issue/topic.\n"
    "- intent_candidate: short noun phrase (e.g., 'Call Flow Configuration', 'Pricing Inquiry').\n"
    "- scenario_candidate: specific variation.\n"
    "- flow_steps_candidate: 3–8 semantic steps (NOT click-by-click UI).\n"
    "- resolution.summary: what happened in THIS conversation.\n\n"
    "5) Blueprint signals:\n"
    "- policy: explicit restrictions/promises mentioned.\n"
    "- escalations: handover patterns (share screenshot, connect AM, raise ticket).\n"
    "- tone: 1–3 descriptors.\n\n"
    f"{ATOMS_SCHEMA}\n\nConversation:\n{{conversation}}"
)

REPAIR_SYSTEM = "Fix JSON formatting only. Always respond in English."
REPAIR_USER = "Return valid JSON only. Do not change meaning.\n{broken_json}"


class AtomsExtractionError(RuntimeError):
    pass


def extract_atoms(
    conversations: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    cache: FileCache,
    system_prompt: str = SYSTEM_PROMPT,
    user_template: str = USER_TEMPLATE,
    repair_system: str = REPAIR_SYSTEM,
    repair_user: str = REPAIR_USER,
    progress_cb=None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    outputs: List[Dict[str, Any]] = []
    failures: List[str] = []

    total = len(conversations)
    for idx, convo in enumerate(conversations, start=1):
        payload = json.dumps(convo, ensure_ascii=True)
        if "{conversation}" in user_template:
            prompt = user_template.replace("{conversation}", payload)
        else:
            prompt = f"{user_template}\n\nConversation:\n{payload}"
        key = cache.key_for(model, system_prompt, prompt)
        cached = cache.get(key)
        if cached is None:
            try:
                parsed = chat_json_with_cache(
                    client,
                    model,
                    system_prompt,
                    prompt,
                    cache,
                    key,
                    repair_system=repair_system,
                    repair_user=repair_user,
                )
                if isinstance(parsed, dict) and not parsed.get("conversation_id"):
                    parsed["conversation_id"] = convo.get("conversation_id", "unknown")
                cached = parsed
            except Exception:  # noqa: BLE001
                failures.append(convo.get("conversation_id", "unknown"))
                continue

        if not cached:
            failures.append(convo.get("conversation_id", "unknown"))
            continue
        try:
            atoms = Atoms(**cached)
            outputs.append(atoms.model_dump())
        except Exception:  # noqa: BLE001
            failures.append(convo.get("conversation_id", "unknown"))
        if progress_cb:
            progress_cb(idx, total)

    return outputs, failures
