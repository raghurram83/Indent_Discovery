from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from ..utils import sha256_text


TOOL_CATALOG = {
    "create_demo_meeting": {
        "description": "Schedule a demo meeting and return confirmation details.",
        "parameters": {
            "type": "object",
            "properties": {"fields": {"type": "object", "additionalProperties": True}},
            "required": ["fields"],
        },
        "return_type": "json",
    },
    "create_support_ticket": {
        "description": "Create a support ticket and return ticket metadata.",
        "parameters": {
            "type": "object",
            "properties": {"fields": {"type": "object", "additionalProperties": True}},
            "required": ["fields"],
        },
        "return_type": "json",
    },
    "send_whatsapp_message": {
        "description": "Send a WhatsApp message and return delivery metadata.",
        "parameters": {
            "type": "object",
            "properties": {"fields": {"type": "object", "additionalProperties": True}},
            "required": ["fields"],
        },
        "return_type": "json",
    },
    "update_crm_record": {
        "description": "Update a CRM record and return record metadata.",
        "parameters": {
            "type": "object",
            "properties": {"fields": {"type": "object", "additionalProperties": True}},
            "required": ["fields"],
        },
        "return_type": "json",
    },
    "unknown_action": {
        "description": "Fallback action used when no specific tool mapping exists.",
        "parameters": {
            "type": "object",
            "properties": {"fields": {"type": "object", "additionalProperties": True}},
            "required": ["fields"],
        },
        "return_type": "json",
    },
}


def infer_tool_name(action_text: str) -> str:
    lowered = (action_text or "").lower()
    if "demo" in lowered or "meeting" in lowered:
        return "create_demo_meeting"
    if "ticket" in lowered or "support" in lowered:
        return "create_support_ticket"
    if "whatsapp" in lowered or "message" in lowered:
        return "send_whatsapp_message"
    if "crm" in lowered or "update" in lowered:
        return "update_crm_record"
    return "unknown_action"


def _required_fields(scenario: Dict[str, Any]) -> List[str]:
    return list(scenario.get("required_fields") or scenario.get("required_inputs") or [])


def _handover_rules(scenario: Dict[str, Any]) -> List[str]:
    return list(scenario.get("handover_rules") or [])


def _flow_steps(scenario: Dict[str, Any]) -> List[Any]:
    return list(scenario.get("flow_template") or [])


def _journey_id(project_id: str, intent_id: str, scenario_name: str, idx: int) -> str:
    seed = f"{intent_id}:{scenario_name}:{idx}"
    return f"{project_id}::{sha256_text(seed)[:12]}"


def _map_steps(flow_steps: Iterable[Any], required_fields: List[str]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    required_iter = iter(required_fields)
    for step in flow_steps:
        if isinstance(step, dict):
            step_type = (step.get("type") or "inform").lower()
            text = step.get("text") or ""
        else:
            step_type = "inform"
            text = str(step)
        payload: Dict[str, Any] = {"type": step_type, "text": text}
        if step_type == "ask":
            payload["expected_field"] = next(required_iter, None)
        elif step_type == "action":
            tool_name = infer_tool_name(text)
            payload["tool_name"] = tool_name
            payload["action_text"] = text
        steps.append(payload)
    return steps


def map_journeys(intent_payload: Dict[str, Any], project_id: str) -> List[Dict[str, Any]]:
    intents = intent_payload.get("intents", []) if isinstance(intent_payload, dict) else []
    journeys: List[Dict[str, Any]] = []
    for intent_idx, intent in enumerate(intents):
        if not isinstance(intent, dict):
            continue
        intent_id = str(intent.get("intent_id") or f"intent_{intent_idx}")
        intent_name = intent.get("intent_name") or f"Intent {intent_idx}"
        definition = intent.get("definition") or ""
        scenarios = intent.get("scenarios") or []
        for scenario_idx, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                continue
            scenario_name = scenario.get("scenario_name") or f"Scenario {scenario_idx + 1}"
            flow_id = scenario.get("flow_id") or scenario.get("scenario_id") or scenario.get("id")
            journey_id = (
                f"{project_id}::{flow_id}" if flow_id else _journey_id(project_id, intent_id, scenario_name, scenario_idx)
            )
            required = _required_fields(scenario)
            steps = _map_steps(_flow_steps(scenario), required)
            journeys.append(
                {
                    "journey_id": journey_id,
                    "journey_name": scenario_name,
                    "intent_id": intent_id,
                    "intent_name": intent_name,
                    "definition": definition,
                    "required_fields": required,
                    "handover_rules": _handover_rules(scenario),
                    "steps": steps,
                }
            )
    return journeys


def map_global_guidelines(blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    guidelines: List[Dict[str, Any]] = []
    tone = blueprint.get("tone_summary") or []
    if tone:
        top_tone = ", ".join([t.get("tone") for t in tone[:3] if t.get("tone")])
        guidelines.append(
            {"id": "global-tone", "scope": "global", "text": f"Use a tone that is: {top_tone}."}
        )
    policy = blueprint.get("policy_summary") or []
    for idx, item in enumerate(policy):
        text = item.get("policy")
        if text:
            guidelines.append(
                {
                    "id": f"global-policy-{idx + 1}",
                    "scope": "global",
                    "text": f"Policy: {text}.",
                }
            )
    escalations = blueprint.get("escalation_patterns") or []
    for idx, item in enumerate(escalations):
        pattern = item.get("pattern")
        if pattern:
            guidelines.append(
                {
                    "id": f"global-escalation-{idx + 1}",
                    "scope": "global",
                    "text": f"Escalate when you detect: {pattern}.",
                }
            )
    guidelines.append(
        {
            "id": "global-clarify",
            "scope": "global",
            "text": "If required information is missing, ask a clear follow-up question before acting.",
        }
    )
    guidelines.append(
        {
            "id": "global-hallucination",
            "scope": "global",
            "text": "Do not invent facts or policies. If unsure, say so.",
        }
    )
    return guidelines


def map_journey_guidelines(
    journeys: Iterable[Dict[str, Any]], blueprint: Dict[str, Any]
) -> List[Dict[str, Any]]:
    guidelines: List[Dict[str, Any]] = []
    escalation_patterns = [p.get("pattern") for p in (blueprint.get("escalation_patterns") or []) if p.get("pattern")]
    for journey in journeys:
        journey_id = journey.get("journey_id")
        required_fields = journey.get("required_fields") or []
        if required_fields:
            guidelines.append(
                {
                    "id": f"{journey_id}-required-fields",
                    "scope": "journey",
                    "journey_id": journey_id,
                    "text": (
                        "Ask for required fields one at a time before taking actions: "
                        + ", ".join(required_fields)
                        + "."
                    ),
                }
            )
        handover_rules = journey.get("handover_rules") or []
        if handover_rules:
            guidelines.append(
                {
                    "id": f"{journey_id}-handover-rules",
                    "scope": "journey",
                    "journey_id": journey_id,
                    "text": "Escalate to a human when: " + "; ".join(handover_rules) + ".",
                }
            )
        guidelines.append(
            {
                "id": f"{journey_id}-agent-request",
                "scope": "journey",
                "journey_id": journey_id,
                "text": "If the user asks to talk to a human, initiate handover.",
            }
        )
        digression_text = (
            "If the user goes off-topic, answer briefly and return to the active step."
        )
        if escalation_patterns:
            digression_text += " Escalate if off-topic content matches escalation patterns."
        guidelines.append(
            {
                "id": f"{journey_id}-digressions",
                "scope": "journey",
                "journey_id": journey_id,
                "text": digression_text,
            }
        )
    return guidelines


def map_tools(journeys: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools: Dict[str, Dict[str, Any]] = {}
    for journey in journeys:
        for step in journey.get("steps", []):
            if step.get("type") != "action":
                continue
            tool_name = step.get("tool_name") or "unknown_action"
            if tool_name in tools:
                continue
            spec = TOOL_CATALOG.get(tool_name, TOOL_CATALOG["unknown_action"])
            tools[tool_name] = {
                "name": tool_name,
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}),
                "return_type": spec.get("return_type", "json"),
            }
    return list(tools.values())
