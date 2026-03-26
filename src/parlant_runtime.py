from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .chatbot_preview.tools import run_action
from .parlant_publisher.client import ParlantClient, ParlantError


def _extract_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    if payload.get("tool_calls"):
        return list(payload.get("tool_calls") or [])
    messages = payload.get("messages") or []
    for message in messages:
        if isinstance(message, dict) and message.get("tool_calls"):
            return list(message.get("tool_calls") or [])
    choices = payload.get("choices") or []
    for choice in choices:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if msg and msg.get("tool_calls"):
            return list(msg.get("tool_calls") or [])
    return []


def _extract_messages(payload: Dict[str, Any]) -> List[str]:
    if not payload:
        return []
    if payload.get("assistant_message"):
        return [str(payload.get("assistant_message"))]
    messages: List[str] = []
    if payload.get("messages"):
        for msg in payload.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" or "role" not in msg:
                content = msg.get("content") or msg.get("text")
                if content:
                    messages.append(str(content))
    if messages:
        return messages
    choices = payload.get("choices") or []
    for choice in choices:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if msg and msg.get("content"):
            messages.append(str(msg.get("content")))
    return messages


class ParlantRuntime:
    def __init__(self, session_store: Dict[str, Any], client: ParlantClient) -> None:
        self.session_store = session_store
        self.client = client

    def _get_session_id(self, project_id: str) -> str:
        key = f"parlant_session_{project_id}"
        session_id = self.session_store.get(key)
        if not session_id:
            session_id = self.client.create_session(project_id)
            self.session_store[key] = session_id
        return str(session_id)

    def send_message(self, project_id: str, user_text: str) -> Tuple[List[str], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "mode_selected": "parlant",
            "session_id": None,
            "tool_calls": [],
            "tool_results": [],
            "raw_turns": [],
        }
        session_id = self._get_session_id(project_id)
        debug["session_id"] = session_id

        try:
            response = self.client.send_message(session_id, user_text)
            debug["raw_turns"].append(response)
        except ParlantError as exc:
            raise ParlantError(f"Parlant message failed: {exc}") from exc

        messages = _extract_messages(response)
        tool_calls = _extract_tool_calls(response)

        while tool_calls:
            tool_results = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = (
                    call.get("name")
                    or call.get("tool_name")
                    or call.get("tool")
                    or call.get("action")
                    or ""
                )
                arguments = call.get("arguments") or call.get("input") or {}
                tool_call, summary = run_action(tool_name, arguments if isinstance(arguments, dict) else {})
                tool_results.append(
                    {
                        "tool_call_id": call.get("id") or call.get("tool_call_id") or tool_name,
                        "name": tool_name,
                        "input": arguments,
                        "output": tool_call.get("output"),
                        "summary": summary,
                    }
                )
            debug["tool_calls"].extend(tool_calls)
            debug["tool_results"].extend(tool_results)
            response = self.client.send_tool_results(session_id, tool_results)
            debug["raw_turns"].append(response)
            messages = _extract_messages(response)
            tool_calls = _extract_tool_calls(response)

        if not messages and debug["tool_results"]:
            messages = [debug["tool_results"][-1].get("summary", "Action completed.")]
        if not messages:
            messages = ["I am not able to respond right now."]
        return messages, debug
