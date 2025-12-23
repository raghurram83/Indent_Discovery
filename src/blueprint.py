from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from .models import BlueprintPersona, EscalationPattern, GapItem, PolicySummary, ToneSummary


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
