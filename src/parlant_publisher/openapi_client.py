from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests


class OpenAPIError(RuntimeError):
    """Raised when OpenAPI discovery or API requests fail."""


class Endpoint:
    def __init__(self, path: str, method: str, op: Dict[str, Any], score: int) -> None:
        self.path = path
        self.method = method
        self.op = op
        self.score = score


class OpenAPIResolver:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def discover(self, resource: str) -> Dict[str, Optional[Endpoint]]:
        resource = resource.lower()
        candidates: Dict[str, List[Endpoint]] = {
            "list": [],
            "get": [],
            "create": [],
            "update": [],
        }
        for path, methods in self.spec.get("paths", {}).items():
            if not isinstance(methods, dict):
                continue
            for method, op in methods.items():
                method_lower = method.lower()
                if method_lower not in ("get", "post", "put", "patch"):
                    continue
                if not isinstance(op, dict):
                    continue
                text = " ".join(
                    [
                        path,
                        op.get("operationId", ""),
                        op.get("summary", ""),
                        " ".join(op.get("tags", [])),
                    ]
                ).lower()
                if resource not in text and resource.rstrip("s") not in text:
                    continue
                endpoint = Endpoint(path, method_lower, op, self._score(path, op, resource))
                if method_lower == "get":
                    if "{" in path:
                        candidates["get"].append(endpoint)
                    else:
                        candidates["list"].append(endpoint)
                elif method_lower == "post":
                    candidates["create"].append(endpoint)
                else:
                    candidates["update"].append(endpoint)
        return {
            key: self._best(candidates[key])
            for key in ("list", "get", "create", "update")
        }

    def _score(self, path: str, op: Dict[str, Any], resource: str) -> int:
        score = 0
        path_lower = path.lower()
        tags = " ".join(op.get("tags", [])).lower()
        operation_id = op.get("operationId", "").lower()
        summary = op.get("summary", "").lower()
        if f"/{resource}" in path_lower or f"/{resource}s" in path_lower:
            score += 4
        if resource in tags:
            score += 3
        if resource in operation_id:
            score += 2
        if resource in summary:
            score += 1
        score += max(0, 3 - path.count("/"))
        return score

    def _best(self, endpoints: List[Endpoint]) -> Optional[Endpoint]:
        if not endpoints:
            return None
        return sorted(endpoints, key=lambda e: e.score, reverse=True)[0]

    def resolve_schema(self, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not schema:
            return None
        if "$ref" not in schema:
            return schema
        ref = schema.get("$ref", "")
        if not ref.startswith("#/components/schemas/"):
            return None
        name = ref.split("/")[-1]
        return self.spec.get("components", {}).get("schemas", {}).get(name)


class OpenAPIClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.spec: Optional[Dict[str, Any]] = None
        self.resolver: Optional[OpenAPIResolver] = None

    def load_openapi(self) -> None:
        for path in ("/openapi.json", "/api/openapi.json"):
            url = f"{self.base_url}{path}"
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException:
                continue
            if not resp.ok:
                continue
            try:
                self.spec = resp.json()
            except ValueError as exc:
                raise OpenAPIError(f"Failed to parse OpenAPI schema from {url}") from exc
            self.resolver = OpenAPIResolver(self.spec)
            return
        raise OpenAPIError("Failed to fetch OpenAPI schema from Parlant.")

    def list_resource(self, resource: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        endpoint = self._resolve_endpoint(resource, "list")
        data = self._request(endpoint, context, query=self._build_query(endpoint, context))
        return extract_list_response(data)

    def create_resource(
        self, resource: str, context: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        endpoint = self._resolve_endpoint(resource, "create")
        payload = filter_payload(endpoint.op, payload, self.resolver)
        data = self._request(endpoint, context, payload=payload)
        return data if isinstance(data, dict) else {}

    def get_request_schema(self, resource: str, action: str) -> Optional[Dict[str, Any]]:
        endpoint = self._resolve_endpoint(resource, action)
        schema = extract_request_schema(endpoint.op)
        if not schema or not self.resolver:
            return schema
        return self.resolver.resolve_schema(schema) or schema

    def _resolve_endpoint(self, resource: str, action: str) -> Endpoint:
        if not self.resolver:
            raise OpenAPIError("OpenAPI resolver is not initialized.")
        endpoints = self.resolver.discover(resource)
        endpoint = endpoints.get(action)
        if endpoint is None:
            raise OpenAPIError(f"No {action} endpoint found for resource: {resource}")
        return endpoint

    def _request(
        self,
        endpoint: Endpoint,
        context: Dict[str, Any],
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = self._format_url(endpoint, context)
        method = endpoint.method
        resp = self.session.request(
            method,
            url,
            json=payload if method in ("post", "put", "patch") else None,
            params=query if method == "get" else None,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise OpenAPIError(f"{method.upper()} {url} failed: {resp.status_code} {resp.text}")
        if not resp.text.strip():
            return None
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    def _format_url(self, endpoint: Endpoint, context: Dict[str, Any]) -> str:
        path = endpoint.path
        for param in self._extract_path_params(path):
            value = self._path_value(param, context)
            if value is None:
                raise OpenAPIError(f"Missing value for path param: {param}")
            path = path.replace(f"{{{param}}}", value)
        return f"{self.base_url}{path}"

    def _extract_path_params(self, path: str) -> List[str]:
        return re.findall(r"{([^}]+)}", path)

    def _path_value(self, name: str, context: Dict[str, Any]) -> Optional[str]:
        key = name.lower()
        if "agent" in key:
            return str(context.get("agent_id")) if context.get("agent_id") else None
        if "journey" in key:
            return str(context.get("journey_id")) if context.get("journey_id") else None
        if "node" in key:
            return str(context.get("node_id")) if context.get("node_id") else None
        if "edge" in key:
            return str(context.get("edge_id")) if context.get("edge_id") else None
        if key in context:
            return str(context[key])
        return None

    def _build_query(self, endpoint: Endpoint, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        params = endpoint.op.get("parameters", []) if isinstance(endpoint.op, dict) else []
        if not isinstance(params, list):
            return None
        query: Dict[str, Any] = {}
        for param in params:
            if not isinstance(param, dict):
                continue
            if param.get("in") != "query":
                continue
            name = param.get("name")
            if not name:
                continue
            if name in context:
                query[name] = context[name]
                continue
            if name.lower() in ("agent_id", "agentid") and context.get("agent_id"):
                query[name] = context.get("agent_id")
        return query or None


def extract_request_schema(op: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_body = op.get("requestBody", {})
    content = request_body.get("content", {})
    if not isinstance(content, dict):
        return None
    for content_type in ("application/json", "application/*+json", "application/json; charset=utf-8"):
        if content_type in content:
            schema = content[content_type].get("schema")
            if isinstance(schema, dict):
                return schema
    for value in content.values():
        if isinstance(value, dict) and isinstance(value.get("schema"), dict):
            return value["schema"]
    return None


def extract_list_response(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "data", "results", "agents", "journeys", "nodes", "edges", "guidelines"):
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if isinstance(item, dict)]
    return []


def extract_id(item: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "agent_id", "journey_id", "uuid"):
        if key in item and item[key]:
            return str(item[key])
    return None


def filter_payload(
    op: Dict[str, Any], payload: Dict[str, Any], resolver: Optional[OpenAPIResolver]
) -> Dict[str, Any]:
    schema = extract_request_schema(op)
    if not schema:
        return payload
    if resolver:
        schema = resolver.resolve_schema(schema) or schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return payload
    return {key: value for key, value in payload.items() if key in properties}


def schema_supports_field(schema: Optional[Dict[str, Any]], field: str) -> bool:
    if not schema or not isinstance(schema, dict):
        return False
    props = schema.get("properties")
    return isinstance(props, dict) and field in props
