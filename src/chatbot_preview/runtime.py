from __future__ import annotations

from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .indexes import search_index
from .parsing import guess_field_value
from .policies import apply_tone, check_escalation
from .tools import run_action


def init_state(state: Dict[str, Any]) -> None:
    state.setdefault("chat_messages", [])
    state.setdefault("active_intent", None)
    state.setdefault("active_scenario", None)
    state.setdefault("step_index", 0)
    state.setdefault("collected_fields", {})
    state.setdefault("pending_ask", None)
    state.setdefault("pending_ask_type", None)
    state.setdefault("last_debug", None)
    state.setdefault("debug_history", [])


def _missing_required_fields(scenario: Dict[str, Any], collected: Dict[str, Any]) -> List[str]:
    required = scenario.get("required_fields") or scenario.get("required_inputs") or []
    missing = []
    for field in required:
        field_name = str(field)
        if field_name and field_name not in collected:
            missing.append(field_name)
    return missing


def _advance_flow(
    scenario: Dict[str, Any],
    state: Dict[str, Any],
    blueprint: Dict[str, Any] | None,
) -> Tuple[List[str], Dict[str, Any]]:
    messages: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    flow_steps = scenario.get("flow_template") or []
    while state["step_index"] < len(flow_steps):
        step = flow_steps[state["step_index"]]
        step_type = ""
        step_text = ""
        if isinstance(step, dict):
            step_type = (step.get("type") or "").lower()
            step_text = step.get("text") or ""
        else:
            step_type = "inform"
            step_text = str(step)
        if step_type == "ask":
            state["pending_ask"] = step_text
            state["pending_ask_type"] = "step"
            state["step_index"] += 1
            messages.append(apply_tone(step_text, blueprint))
            break
        if step_type == "handover":
            state["pending_ask"] = None
            state["pending_ask_type"] = None
            messages.append(apply_tone("I am connecting you with a human specialist.", blueprint))
            state["active_intent"] = None
            state["active_scenario"] = None
            state["step_index"] = 0
            state["collected_fields"] = {}
            break
        if step_type == "action":
            missing = _missing_required_fields(scenario, state["collected_fields"])
            if missing:
                ask_text = f"Could you share your {missing[0]}?"
                state["pending_ask"] = ask_text
                state["pending_ask_type"] = "field"
                messages.append(apply_tone(ask_text, blueprint))
                break
            tool_call, summary = run_action(step_text, state["collected_fields"])
            tool_calls.append(tool_call)
            messages.append(apply_tone(summary, blueprint))
            state["step_index"] += 1
            continue
        messages.append(apply_tone(step_text, blueprint))
        state["step_index"] += 1
    if state["step_index"] >= len(flow_steps) and state["active_intent"]:
        messages.append(apply_tone("That is complete. Let me know if you need anything else.", blueprint))
        state["active_intent"] = None
        state["active_scenario"] = None
        state["step_index"] = 0
        state["collected_fields"] = {}
        state["pending_ask"] = None
        state["pending_ask_type"] = None
    return messages, {"tool_calls": tool_calls}


def _select_mode(
    user_msg: str,
    faq_matches: List[Dict[str, Any]],
    scenario_matches: List[Dict[str, Any]],
    faq_threshold: float,
    scenario_threshold: float,
) -> str:
    if faq_matches:
        top_faq = faq_matches[0]
        faq_item = top_faq.get("item") or {}
        confidence = (faq_item.get("confidence") or "").lower()
        if top_faq.get("score", 0) >= faq_threshold and confidence != "low":
            return "faq"
    if scenario_matches and scenario_matches[0].get("score", 0) >= scenario_threshold:
        return "flow"
    return "fallback"


def _has_pricing_intent(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in ["price", "pricing", "cost", "fee", "charge", "plan", "billing"])


def _render_prompt(
    template: str,
    user_msg: str,
    mode: str,
    base_response: str,
    faq_matches: List[Dict[str, Any]],
    scenario_matches: List[Dict[str, Any]],
    flow_state: Dict[str, Any],
    blueprint: Dict[str, Any] | None,
) -> str:
    payload = {
        "user_message": user_msg,
        "mode_selected": mode,
        "base_response": base_response,
        "faq_matches": faq_matches,
        "scenario_matches": scenario_matches,
        "flow_state": flow_state,
        "blueprint": blueprint or {},
    }
    prompt = template or ""
    for key, value in payload.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt


def _llm_response(client: OpenAI, model: str, prompt: str) -> str:
    if not prompt:
        return ""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a chatbot preview."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def handle_user_message(
    user_msg: str,
    assets: Dict[str, Any],
    client: OpenAI,
    embed_model: str,
    faq_threshold: float,
    scenario_threshold: float,
    use_llm: bool,
    llm_model: str,
    prompt_template: str,
    state: Dict[str, Any],
    blueprint: Dict[str, Any] | None,
) -> Tuple[List[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "mode_selected": None,
        "top_faq_matches": [],
        "top_scenario_matches": [],
        "active_flow_state": {},
        "policy_decisions": {"escalated": False, "reason": ""},
        "tool_calls": [],
    }
    escalation = check_escalation(user_msg, blueprint)
    if escalation.get("escalate"):
        debug["mode_selected"] = "fallback"
        debug["policy_decisions"] = {"escalated": True, "reason": escalation.get("reason", "")}
        state["active_intent"] = None
        state["active_scenario"] = None
        state["step_index"] = 0
        state["pending_ask"] = None
        state["pending_ask_type"] = None
        response = apply_tone("I am connecting you with a human specialist for this request.", blueprint)
        return [response], debug

    if state.get("pending_ask"):
        scenario = state.get("active_scenario")
        if scenario:
            scenario_item = scenario
            missing = _missing_required_fields(scenario_item, state["collected_fields"])
            if missing:
                field_name = missing[0]
                state["collected_fields"][field_name] = guess_field_value(user_msg, field_name)
            else:
                state["collected_fields"]["last_response"] = user_msg.strip()
        state["pending_ask"] = None
        state["pending_ask_type"] = None
        messages, tool_meta = _advance_flow(state.get("active_scenario") or {}, state, blueprint)
        debug["mode_selected"] = "flow"
        debug["tool_calls"] = tool_meta.get("tool_calls", [])
        debug["active_flow_state"] = {
            "active_intent": state.get("active_intent"),
            "active_scenario": (state.get("active_scenario") or {}).get("scenario_name"),
            "step_index": state.get("step_index"),
            "collected_fields": dict(state.get("collected_fields") or {}),
            "pending_ask": state.get("pending_ask"),
        }
        if use_llm and messages:
            flow_state = debug["active_flow_state"]
            prompt = _render_prompt(
                prompt_template,
                user_msg,
                debug["mode_selected"],
                "\n".join(messages),
                [],
                [],
                flow_state,
                blueprint,
            )
            debug["prompt_rendered"] = prompt
            llm_text = _llm_response(client, llm_model, prompt)
            if llm_text:
                return [llm_text], debug
        return messages, debug

    faq_index = assets.get("faq_index")
    scenario_index = assets.get("scenario_index")
    faq_matches = search_index(user_msg, faq_index, client, embed_model, top_k=5) if faq_index else []
    scenario_matches = (
        search_index(user_msg, scenario_index, client, embed_model, top_k=5) if scenario_index else []
    )

    debug["top_faq_matches"] = [
        {
            "faq_id": (m.get("item") or {}).get("faq_id"),
            "canonical_question": (m.get("item") or {}).get("canonical_question"),
            "score": m.get("score"),
            "details": m.get("item"),
        }
        for m in faq_matches
    ]
    debug["top_scenario_matches"] = [
        {
            "intent_name": (m.get("item") or {}).get("intent_name"),
            "scenario_name": (m.get("item") or {}).get("scenario_name"),
            "score": m.get("score"),
            "details": m.get("item"),
        }
        for m in scenario_matches
    ]

    mode = _select_mode(user_msg, faq_matches, scenario_matches, faq_threshold, scenario_threshold)
    if mode == "fallback" and _has_pricing_intent(user_msg):
        if faq_matches:
            mode = "faq"
        elif scenario_matches:
            mode = "flow"
    debug["mode_selected"] = mode

    if mode == "faq":
        top_faq = faq_matches[0]["item"]
        answer = (top_faq.get("answer") or "").strip()
        if not answer:
            answer = "I found a related FAQ, but the answer is missing."
        if top_faq.get("needs_verification"):
            answer = f"{answer} Please verify before acting."
        base = apply_tone(answer, blueprint)
        if use_llm:
            prompt = _render_prompt(
                prompt_template,
                user_msg,
                mode,
                base,
                debug["top_faq_matches"],
                debug["top_scenario_matches"],
                {},
                blueprint,
            )
            debug["prompt_rendered"] = prompt
            llm_text = _llm_response(client, llm_model, prompt)
            if llm_text:
                return [llm_text], debug
        return [base], debug

    if mode == "flow":
        best = scenario_matches[0]["item"]
        state["active_intent"] = best.get("intent_name")
        state["active_scenario"] = best
        state["step_index"] = 0
        state["collected_fields"] = {}
        state["pending_ask"] = None
        state["pending_ask_type"] = None
        messages, tool_meta = _advance_flow(best, state, blueprint)
        debug["tool_calls"] = tool_meta.get("tool_calls", [])
        debug["active_flow_state"] = {
            "active_intent": state.get("active_intent"),
            "active_scenario": best.get("scenario_name"),
            "step_index": state.get("step_index"),
            "collected_fields": dict(state.get("collected_fields") or {}),
            "pending_ask": state.get("pending_ask"),
        }
        if use_llm and messages:
            prompt = _render_prompt(
                prompt_template,
                user_msg,
                mode,
                "\n".join(messages),
                debug["top_faq_matches"],
                debug["top_scenario_matches"],
                debug["active_flow_state"],
                blueprint,
            )
            debug["prompt_rendered"] = prompt
            llm_text = _llm_response(client, llm_model, prompt)
            if llm_text:
                return [llm_text], debug
        return messages, debug

    options = []
    if faq_matches:
        options.append((faq_matches[0]["item"] or {}).get("canonical_question"))
    if scenario_matches:
        options.append((scenario_matches[0]["item"] or {}).get("scenario_name"))
    options = [o for o in options if o]
    if _has_pricing_intent(user_msg) and not options:
        clarifier = (
            "I do not have pricing details in the loaded FAQs. "
            "Please upload a pricing FAQ or share the plan name you want pricing for."
        )
    elif options:
        clarifier = "Could you clarify if you meant: " + "; ".join(options[:3]) + "?"
    else:
        clarifier = "Could you clarify what you need help with?"
    base = apply_tone(clarifier, blueprint)
    if use_llm:
        prompt = _render_prompt(
            prompt_template,
            user_msg,
            mode,
            base,
            debug["top_faq_matches"],
            debug["top_scenario_matches"],
            {},
            blueprint,
        )
        debug["prompt_rendered"] = prompt
        llm_text = _llm_response(client, llm_model, prompt)
        if llm_text:
            return [llm_text], debug
    return [base], debug
