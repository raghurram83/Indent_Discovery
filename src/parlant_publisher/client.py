from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, List

import requests


class ParlantError(RuntimeError):
    """Raised when Parlant API requests fail."""


DEFAULT_ENDPOINTS: Dict[str, Any] = {
    "health": ["/health", "/api/health", "/status"],
    "journeys": "/api/journeys",
    "guidelines": "/api/guidelines",
    "tools": "/api/tools",
    "sessions": "/api/sessions",
    "session_messages": "/api/sessions/{session_id}/messages",
    "session_tool_results": "/api/sessions/{session_id}/tool_results",
}


class ParlantClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        retries: int = 2,
        session: Optional[requests.Session] = None,
        endpoints: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self.session = session or requests.Session()
        self.endpoints = dict(DEFAULT_ENDPOINTS)
        if endpoints:
            self.endpoints.update(endpoints)

    def _request(
        self,
        method: str,
        path: str,
        json_payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        expect_json: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    params=params,
                    timeout=self.timeout,
                )
                if resp.status_code >= 500 and attempt < self.retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                if not resp.ok:
                    raise ParlantError(f"{method} {url} failed: {resp.status_code} {resp.text}")
                if not expect_json:
                    return resp.text
                if not resp.text.strip():
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    return {"raw": resp.text}
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise ParlantError(f"{method} {url} failed: {exc}") from exc
        if last_error:
            raise ParlantError(f"{method} {url} failed: {last_error}") from last_error
        raise ParlantError(f"{method} {url} failed: unknown error")

    def check_connection(self) -> bool:
        for path in self.endpoints.get("health", []):
            try:
                self._request("GET", path, expect_json=False)
                return True
            except ParlantError:
                continue
        try:
            self._request("GET", "", expect_json=False)
            return True
        except ParlantError:
            return False

    def auto_configure(self) -> Dict[str, Any]:
        candidates = ["/openapi.json", "/api/openapi.json"]
        for path in candidates:
            try:
                resp = self.session.get(f"{self.base_url}{path}", timeout=self.timeout)
            except requests.RequestException:
                continue
            if not resp.ok:
                continue
            try:
                payload = resp.json()
            except ValueError:
                continue
            paths = payload.get("paths") or {}
            if not isinstance(paths, dict):
                continue
            updates: Dict[str, str] = {}
            path_keys = list(paths.keys())

            def pick(includes: List[str], excludes: List[str] | None = None, longest: bool = False) -> str | None:
                excludes = excludes or []
                matches = [
                    p
                    for p in path_keys
                    if all(token in p.lower() for token in includes)
                    and not any(token in p.lower() for token in excludes)
                ]
                if not matches:
                    return None
                return sorted(matches, key=len, reverse=longest)[0]

            journeys_path = pick(["journey"])
            guidelines_path = pick(["guideline"])
            tools_path = pick(["tool"], excludes=["result", "call", "session", "message"])
            sessions_path = pick(["session"], excludes=["message", "tool"], longest=False)
            session_messages_path = pick(["session", "message"], longest=True) or pick(
                ["session", "chat"], longest=True
            )
            session_tools_path = pick(["tool", "result"], longest=True) or pick(
                ["session", "tool"], longest=True
            )

            if journeys_path:
                updates["journeys"] = journeys_path
            if guidelines_path:
                updates["guidelines"] = guidelines_path
            if tools_path:
                updates["tools"] = tools_path
            if sessions_path:
                updates["sessions"] = sessions_path
            if session_messages_path:
                updates["session_messages"] = session_messages_path
            if session_tools_path:
                updates["session_tool_results"] = session_tools_path

            if updates:
                self.endpoints.update(updates)
                return {"ok": True, "source": path, "updates": updates, "paths": path_keys}
            return {"ok": False, "source": path, "paths": path_keys}
        return {"ok": False}

    def upsert_journey(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", self.endpoints["journeys"], json_payload=payload)

    def upsert_guideline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", self.endpoints["guidelines"], json_payload=payload)

    def upsert_tool(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", self.endpoints["tools"], json_payload=payload)

    def create_session(self, project_id: str) -> str:
        payload = {"project_id": project_id}
        response = self._request("POST", self.endpoints["sessions"], json_payload=payload)
        session_id = response.get("session_id") or response.get("id")
        if not session_id:
            raise ParlantError("Parlant did not return a session id.")
        return str(session_id)

    def send_message(self, session_id: str, user_text: str) -> Dict[str, Any]:
        payload = {"message": user_text, "input": user_text}
        path = self.endpoints["session_messages"].format(session_id=session_id)
        return self._request("POST", path, json_payload=payload)

    def send_tool_results(self, session_id: str, tool_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"tool_results": list(tool_results)}
        path = self.endpoints["session_tool_results"].format(session_id=session_id)
        return self._request("POST", path, json_payload=payload)
