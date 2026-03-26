from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Dict, List

from openai import OpenAI

from .models import BlueprintPersona, EscalationPattern, GapItem, PolicySummary, ToneSummary
from .cache import FileCache
from .openai_client import chat_json_with_cache


SYSTEM_PROMPT = "You are an analyst who produces a business blueprint summary in JSON only."
USER_TEMPLATE = (
    "Given the base blueprint and evidence, return JSON only with keys:\n"
    "tone_summary (list of {tone, count}), policy_summary (list of {policy, count, examples}),\n"
    "escalation_patterns (list of {pattern, count, examples}), gaps (list of {faq_id, reason}).\n"
    "Do not add keys. Use the same schema.\n\n"
    "Base blueprint:\n{base_blueprint}\n\n"
    "Atoms rows:\n{atoms_rows}\n\n"
    "FAQ entries:\n{faq_entries}\n"
)


def _top_examples(values: List[str], limit: int = 3) -> List[str]:
    return values[:limit]


def build_blueprint(atoms_rows: List[Dict[str, Any]], faq_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    tone_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()
    escalation_counter: Counter[str] = Counter()

    policy_examples: Dict[str, List[str]] = defaultdict(list)
    escalation_examples: Dict[str, List[str]] = defaultdict(list)

    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id", "unknown")
        blueprint = atoms.get("blueprint_signals") or {}
        tone = blueprint.get("tone") or []
        policy = blueprint.get("policy") or []
        escalations = blueprint.get("escalations") or []

        tone_counter.update(tone)
        policy_counter.update(policy)
        escalation_counter.update(escalations)

        for item in policy:
            policy_examples[item].append(convo_id)
        for item in escalations:
            escalation_examples[item].append(convo_id)

    tone_summary = [ToneSummary(tone=k, count=v) for k, v in tone_counter.most_common()]
    policy_summary = [
        PolicySummary(policy=k, count=v, examples=_top_examples(policy_examples.get(k, [])))
        for k, v in policy_counter.most_common()
    ]
    escalation_patterns = [
        EscalationPattern(pattern=k, count=v, examples=_top_examples(escalation_examples.get(k, [])))
        for k, v in escalation_counter.most_common()
    ]

    gaps: List[GapItem] = []
    for faq in faq_entries:
        variants = faq.get("question_variants") or []
        freq = len(variants)
        draft = faq.get("draft_answer") or ""
        needs_verification = bool(faq.get("needs_verification"))
        if freq >= 3 and (needs_verification or "NEEDS_INFO" in draft):
            gaps.append(GapItem(faq_id=faq.get("faq_id", "unknown"), reason="needs_more_answer_detail"))

    blueprint = BlueprintPersona(
        tone_summary=tone_summary,
        policy_summary=policy_summary,
        escalation_patterns=escalation_patterns,
        gaps=gaps,
    )
    return blueprint.model_dump()


def build_blueprint_with_llm(
    atoms_rows: List[Dict[str, Any]],
    faq_entries: List[Dict[str, Any]],
    client: OpenAI,
    llm_model: str,
    llm_cache: FileCache,
    system_prompt: str = SYSTEM_PROMPT,
    user_template: str = USER_TEMPLATE,
) -> Dict[str, Any]:
    base = build_blueprint(atoms_rows, faq_entries)
    prompt = user_template
    prompt = prompt.replace("{base_blueprint}", json.dumps(base, ensure_ascii=True))
    prompt = prompt.replace("{atoms_rows}", json.dumps(atoms_rows, ensure_ascii=True))
    prompt = prompt.replace("{faq_entries}", json.dumps(faq_entries, ensure_ascii=True))
    cache_key = llm_cache.key_for(llm_model, system_prompt, prompt)
    parsed = chat_json_with_cache(client, llm_model, system_prompt, prompt, llm_cache, cache_key)
    if isinstance(parsed, dict) and parsed:
        return parsed
    return base
