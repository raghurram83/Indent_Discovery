from __future__ import annotations

from typing import Any, Dict


def apply_tone(text: str, blueprint: Dict[str, Any] | None) -> str:
    if not text:
        return text
    if not blueprint:
        return text
    tone_summary = blueprint.get("tone_summary") or []
    tone_text = " ".join([str(t) for t in tone_summary]).lower()
    if "formal" in tone_text:
        return f"Certainly. {text}"
    if "friendly" in tone_text:
        return f"Sure. {text}"
    if "concise" in tone_text:
        return text
    return text


def check_escalation(user_msg: str, blueprint: Dict[str, Any] | None) -> Dict[str, Any]:
    if not blueprint or not user_msg:
        return {"escalate": False, "reason": ""}
    msg = user_msg.lower()
    patterns = blueprint.get("escalation_patterns") or []
    for pattern in patterns:
        if not pattern:
            continue
        if str(pattern).lower() in msg:
            return {"escalate": True, "reason": f"Matched escalation pattern: {pattern}"}
    policy_summary = blueprint.get("policy_summary") or []
    for policy in policy_summary:
        if not policy:
            continue
        policy_text = str(policy).lower()
        if "escalate" in policy_text or "handover" in policy_text:
            if "refund" in msg or "legal" in msg or "compliance" in msg:
                return {"escalate": True, "reason": f"Policy trigger: {policy}"}
    return {"escalate": False, "reason": ""}
