from __future__ import annotations

import io
import zipfile
import os
import re
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from shutil import rmtree
import requests

from src.atoms import SYSTEM_PROMPT as ATOMS_SYSTEM_PROMPT
from src.atoms import USER_TEMPLATE as ATOMS_USER_TEMPLATE
from src.atoms import extract_atoms
from src.blueprint import SYSTEM_PROMPT as BLUEPRINT_SYSTEM_PROMPT
from src.blueprint import USER_TEMPLATE as BLUEPRINT_USER_TEMPLATE
from src.blueprint import build_blueprint, build_blueprint_with_llm
from src.cache import FileCache
from src.clustering import cluster_embeddings, split_large_clusters
from src.embeddings import embed_texts
from src.faq import SYSTEM_PROMPT as FAQ_SYSTEM_PROMPT
from src.faq import USER_TEMPLATE as FAQ_USER_TEMPLATE
from src.faq import build_faq_catalogue, build_faq_from_cluster, cluster_faq_rows, collect_faq_rows
from src.importer import normalize_records, read_raw_records
from src.intent_flow import SYSTEM_PROMPT as INTENT_SYSTEM_PROMPT
from src.intent_flow import USER_TEMPLATE as INTENT_USER_TEMPLATE
from src.intent_flow import build_intent_flow_catalogue, build_intent_flow_catalogue_from_clusters
from src.storage import get_workspace
from src.utils import ensure_dir, load_json, load_jsonl, save_json, save_jsonl, sha256_text
from src.chatbot_preview.indexes import build_faq_index, build_scenario_index
from src.chatbot_preview.runtime import handle_user_message, init_state
from src.parlant_publisher.client import ParlantClient, ParlantError
from src.parlant_publisher.openapi_client import (
    OpenAPIClient,
    OpenAPIError,
    extract_id,
    schema_supports_field,
)
from src.parlant_publisher.sync import sync_project_to_parlant
from src.parlant_runtime import ParlantRuntime
from src.openai_client import call_chat, chat_json_with_cache, with_retries


APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT / "workspace"

st.set_page_config(page_title="Conversation-Driven Intent/FAQ/Blueprint", layout="wide")

PARLANT_JOURNEY_SYSTEM_PROMPT = (
    "You are GPT-5.2 acting as a deterministic Conversation-to-Journey compiler. "
    "Return JSON only."
)

PARLANT_JOURNEY_USER_TEMPLATE = (
    "You are a deterministic Conversation-to-Journey compiler.\n"
    "\n"
    "Your task is to convert structured intent + scenario definitions into\n"
    "Parlant-compatible journeys.\n"
    "\n"
    "You MUST:\n"
    "- Make NO assumptions\n"
    "- Use ONLY the provided input\n"
    "- Preserve structure and intent faithfully\n"
    "- Produce output that is directly pushable to Parlant\n"
    "- Use only these fields: {allowed_fields}\n"
    "- Required fields: {required_fields}\n"
    "- Include this key for dedupe in conditions: [key={key}]\n"
    "\n"
    "You MUST NOT:\n"
    "- Ask clarifying questions\n"
    "- Invent business logic\n"
    "- Invent API behavior\n"
    "- Invent validation rules\n"
    "- Merge scenarios\n"
    "- Modify step order\n"
    "\n"
    "---\n"
    "\n"
    "INPUT\n"
    "You will receive a JSON object with:\n"
    "- intent_id\n"
    "- intent_name\n"
    "- definition\n"
    "- scenario (contains flow_template, required_fields, handover_rules, etc.)\n"
    "\n"
    "---\n"
    "\n"
    "TARGET OUTPUT: PARLANT JOURNEY (single object)\n"
    "\n"
    "Journey Rules\n"
    "1) One scenario = one journey\n"
    "2) Keep scenario intent exactly\n"
    "3) Steps must follow flow_template order\n"
    "4) Use neutral, business-safe tone; no emojis\n"
    "\n"
    "OUTPUT FORMAT (STRICT)\n"
    "Return a single JSON object using only the allowed fields and required fields.\n"
    "Output JSON ONLY (no explanations).\n"
    "\n"
    "Intent JSON:\n{intent_json}\n\n"
    "Scenario JSON:\n{scenario_json}\n"
)

PARLANT_CLUSTER_SYSTEM_PROMPT = (
    "You are a deterministic compiler that converts conversation atoms into Parlant-ready payloads. "
    "Return JSON only. No explanations."
)

PARLANT_CLUSTER_USER_TEMPLATE = (
    "CONTEXT\n"
    "Parlant uses:\n"
    "- Journeys: structured flows that guide users step-by-step.\n"
    "- Guidelines: concise rules/answers that steer behavior.\n"
    "- Journey nodes/edges: explicit state machine steps and transitions.\n"
    "\n"
    "TASK\n"
    "You will receive a JSON array of conversation atoms for a single cluster.\n"
    "Generate Parlant candidates from ONLY this input.\n"
    "\n"
    "OUTPUT (JSON only)\n"
    "{\n"
    '  "journeys": [ ... ],\n'
    '  "guidelines": [ ... ],\n'
    '  "journey_nodes": [ ... ],\n'
    '  "journey_edges": [ ... ]\n'
    "}\n"
    "\n"
    "FIELD CONSTRAINTS (MANDATORY)\n"
    "- Use ONLY these journey fields: {journey_fields}\n"
    "- Required journey fields: {journey_required}\n"
    "- Use ONLY these guideline fields: {guideline_fields}\n"
    "- Required guideline fields: {guideline_required}\n"
    "- Use ONLY these node fields: {node_fields}\n"
    "- Required node fields: {node_required}\n"
    "- Use ONLY these edge fields: {edge_fields}\n"
    "- Required edge fields: {edge_required}\n"
    "\n"
    "HARD RULES\n"
    "- Use ONLY the atoms. Do NOT invent facts, steps, or policies.\n"
    "- Do NOT merge unrelated scenarios. One scenario -> one journey.\n"
    "- Preserve order of flow steps if present.\n"
    "- Keep output short, literal, and deterministic.\n"
    "- journey_nodes and journey_edges MUST be present and non-empty.\n"
    "\n"
    "JOURNEY NODES & EDGES\n"
    "- Each journey_nodes item must include journey_title and a node label.\n"
    "- Each journey_edges item must include journey_title, source_label, target_label.\n"
    "- Use labels to connect edges to nodes within the same journey_title.\n"
    "- Node action should be the assistant message for that step.\n"
    "- Edge condition should describe the user signal that moves to the next step.\n"
    "- If the cluster includes pricing/plan/quote disputes, build a multi-step pricing inquiry flow.\n"
    "\n"
    "GUIDELINE MAPPING\n"
    "- FAQ guideline: condition = faq_candidates.q_clean; action = best non-NEEDS_INFO a_candidate.\n"
    "- Tone guideline: use blueprint_signals.tone (single guideline).\n"
    "- Escalation guideline: use blueprint_signals.escalations (single guideline).\n"
    "\n"
    "RETURN JSON ONLY.\n"
    "\n"
    "Atoms JSON:\n{atoms_json}\n"
)

FINAL_PROMPT_SYSTEM = "You are a senior prompt engineer. Return only the system prompt text."

FINAL_PROMPT_USER_TEMPLATE = (
    "You are given an intent JSON describing an automation-ready intent and its scenarios.\n"
    "Write a SINGLE system prompt for a production chatbot that can automate this intent end-to-end.\n"
    "\n"
    "Requirements:\n"
    "- Cover all scenarios included in the intent.\n"
    "- Be explicit about required fields and how to ask for them.\n"
    "- Include escalation/handover conditions if present.\n"
    "- Keep tone professional, concise, and deterministic.\n"
    "- Output ONLY the system prompt text. No JSON, no markdown.\n"
    "\n"
    "Intent JSON:\n{intent_json}\n"
)

JUDGE_SYSTEM_PROMPT = "You are a strict evaluator. Return JSON only."
JUDGE_USER_TEMPLATE = (
    "Evaluate the chatbot using the intent specification and transcripts.\n"
    "Score each category from 0 to 5 (5 is best).\n"
    "\n"
    "Categories:\n"
    "- task_completion: did the bot resolve the intent correctly?\n"
    "- required_fields: did it collect required inputs appropriately?\n"
    "- flow_adherence: did it follow the intended flow and order?\n"
    "- response_quality: clarity, tone, and helpfulness.\n"
    "- escalation_handling: correct handover/escalation behavior if applicable.\n"
    "\n"
    "If a category is not applicable, score based on correct handling (e.g., 5 if no escalation was needed and none was triggered).\n"
    "Return ONLY JSON in this schema:\n"
    "{{\n"
    '  "scores": {{\n'
    '    "task_completion": 0,\n'
    '    "required_fields": 0,\n'
    '    "flow_adherence": 0,\n'
    '    "response_quality": 0,\n'
    '    "escalation_handling": 0\n'
    "  }},\n"
    '  "rationales": {{\n'
    '    "task_completion": "...",\n'
    '    "required_fields": "...",\n'
    '    "flow_adherence": "...",\n'
    '    "response_quality": "...",\n'
    '    "escalation_handling": "..."\n'
    "  }},\n"
    '  "overall_notes": "..."\n'
    "}}\n"
    "\n"
    "Intent JSON:\n{intent_json}\n\n"
    "System prompt:\n{system_prompt}\n\n"
    "Original transcript:\n{original_transcript}\n\n"
    "Simulated transcript:\n{simulated_transcript}\n"
)


def list_projects() -> List[str]:
    if not WORKSPACE_ROOT.exists():
        return []
    return sorted([p.name for p in WORKSPACE_ROOT.iterdir() if p.is_dir()])


def get_openai_client(api_key: str) -> OpenAI | None:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_parlant_client(base_url: str) -> ParlantClient | None:
    if not base_url:
        return None
    return ParlantClient(base_url)


def test_parlant_connection(base_url: str, timeout_s: float = 2.5) -> Dict[str, Any]:
    if not base_url:
        return {"ok": False, "error": "Parlant base URL is required."}
    url = base_url.rstrip("/") + "/"
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "error": "Parlant server not running. Start with: python parlant_server/main.py",
            "details": str(exc),
        }
    if 200 <= response.status_code < 400:
        return {"ok": True, "status": response.status_code}
    health_paths = ["api/health", "health", "docs", "openapi.json", "redoc"]
    for path in health_paths:
        try:
            health_url = url.rstrip("/") + "/" + path
            health_resp = requests.get(health_url, timeout=timeout_s)
            if 200 <= health_resp.status_code < 400:
                return {
                    "ok": True,
                    "status": health_resp.status_code,
                    "warning": (
                        f"Root path returned {response.status_code}. "
                        "Server is reachable via health check."
                    ),
                }
        except requests.RequestException:
            continue
    if response.status_code in {401, 403, 404, 405}:
        return {
            "ok": True,
            "status": response.status_code,
            "warning": (
                f"Parlant server is reachable but returned {response.status_code} at the base URL. "
                "Check the base URL path."
            ),
        }
    return {
        "ok": False,
        "error": f"Parlant server responded with status {response.status_code}.",
        "details": response.text[:200],
    }


def fetch_parlant_agents(base_url: str) -> Dict[str, Any]:
    client = OpenAPIClient(base_url)
    client.load_openapi()
    agents = client.list_resource("agent", {})
    return {"client": client, "agents": agents}


def build_journey_payload(
    intent: Dict[str, Any],
    scenario: Dict[str, Any],
    key: str,
    schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    intent_name = intent.get("intent_name") or "Unknown intent"
    scenario_name = scenario.get("scenario_name") or "Scenario"
    definition = intent.get("definition") or ""
    flow_steps = scenario.get("flow_template") or []
    flow_text = " -> ".join([step.get("text", "") for step in flow_steps if isinstance(step, dict)])
    description_lines = [
        f"Intent: {intent_name}",
        f"Scenario: {scenario_name}",
    ]
    if definition:
        description_lines.append(f"Definition: {definition}")
    if flow_text:
        description_lines.append(f"Flow: {flow_text}")
    conditions = [
        f"customer intent is {intent_name}",
        f"customer scenario is {scenario_name}",
        f"[key={key}]",
    ]
    payload = {
        "title": scenario_name,
        "description": "\n".join(description_lines),
        "conditions": conditions,
    }
    if schema_supports_field(schema, "tags"):
        payload["tags"] = [
            f"intent:{intent.get('intent_id')}",
            f"scenario:{scenario.get('scenario_id')}",
            f"key:{key}",
        ]
    if schema_supports_field(schema, "id"):
        payload["id"] = f"{intent.get('intent_id')}:{scenario.get('scenario_id')}"
    if schema_supports_field(schema, "name"):
        payload["name"] = scenario_name
    return payload


def filter_payload_by_schema(payload: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not schema or not isinstance(schema, dict):
        return payload
    props = schema.get("properties")
    if not isinstance(props, dict):
        return payload
    return {key: value for key, value in payload.items() if key in props}


def safe_format(template: str, **kwargs: Any) -> str:
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unknown"


def normalize_title(title: str) -> str:
    return title[:100] if len(title) > 100 else title


def build_mermaid_previews(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, str]:
    diagrams: Dict[str, str] = {}
    nodes_by_journey: Dict[str, List[Dict[str, Any]]] = {}
    edges_by_journey: Dict[str, List[Dict[str, Any]]] = {}
    for entry in nodes:
        title = str(entry.get("journey_title") or "journey")
        nodes_by_journey.setdefault(title, []).append(entry)
    for entry in edges:
        title = str(entry.get("journey_title") or "journey")
        edges_by_journey.setdefault(title, []).append(entry)

    for title, journey_nodes in nodes_by_journey.items():
        journey_edges = edges_by_journey.get(title, [])
        label_to_id = {}
        for idx, node in enumerate(journey_nodes, start=1):
            payload = node.get("payload") or {}
            action = payload.get("action") or ""
            raw_label = node.get("label") or slugify(str(action)) or f"node_{idx}"
            label = str(raw_label)[:40]
            label_to_id[label] = f"N{idx}"
        lines = ["stateDiagram-v2", "    direction TB"]
        for node in journey_nodes:
            payload = node.get("payload") or {}
            action = payload.get("action") or ""
            raw_label = node.get("label") or slugify(str(action)) or "node"
            label = str(raw_label)[:40]
            node_id = label_to_id.get(label)
            if not node_id:
                continue
            safe_label = str(action or label).replace("\n", " ").replace('"', "'")
            if len(safe_label) > 80:
                safe_label = safe_label[:77] + "..."
            lines.append(f'    state "{safe_label}" as {node_id}')
        for edge in journey_edges:
            src_label = str(edge.get("source_label") or "")
            tgt_label = str(edge.get("target_label") or "")
            src = label_to_id.get(src_label)
            tgt = label_to_id.get(tgt_label)
            if not src or not tgt:
                continue
            condition = edge.get("payload") or {}
            cond_text = condition.get("condition") or ""
            cond_text = str(cond_text).replace("\n", " ")
            if len(cond_text) > 80:
                cond_text = cond_text[:77] + "..."
            if cond_text:
                lines.append(f"    {src} --> {tgt}: {cond_text}")
            else:
                lines.append(f"    {src} --> {tgt}")
        diagrams[title] = "\n".join(lines)
    return diagrams


def render_mermaid(diagram: str, height: int = 900) -> None:
    uid = sha256_text(diagram)[:10]
    container_id = f"mermaid_{uid}"
    zoom_in_id = f"mermaid_zoom_in_{uid}"
    zoom_out_id = f"mermaid_zoom_out_{uid}"
    zoom_reset_id = f"mermaid_zoom_reset_{uid}"
    html = f"""
<div style="display:flex; gap:8px; margin:6px 0 10px 0;">
  <button id="{zoom_in_id}">Zoom In</button>
  <button id="{zoom_out_id}">Zoom Out</button>
  <button id="{zoom_reset_id}">Reset</button>
  <span style="color:#666; font-size:12px;">Scroll to zoom · drag to pan</span>
</div>
<div id="{container_id}" class="mermaid" style="width:100%; height:{height}px; border:1px solid #e6e1d6; border-radius:8px; overflow:hidden;">
{diagram}
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  (function() {{
    const container = document.getElementById("{container_id}");
    if (!container) return;
    mermaid.initialize({{ startOnLoad: false, flowchart: {{ curve: 'linear' }} }});
    mermaid.init(undefined, container);

    const setup = () => {{
      const svg = container.querySelector("svg");
      if (!svg) return;
      const g = svg.querySelector("g");
      if (!g) return;
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", "100%");
      svg.style.cursor = "grab";

      let scale = 1.2;
      let x = 0;
      let y = 0;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      const update = () => {{
        g.setAttribute("transform", "translate(" + x + ", " + y + ") scale(" + scale + ")");
      }};

      const zoom = (delta) => {{
        const next = Math.min(5, Math.max(0.2, scale + delta));
        scale = next;
        update();
      }};

      document.getElementById("{zoom_in_id}")?.addEventListener("click", () => zoom(0.2));
      document.getElementById("{zoom_out_id}")?.addEventListener("click", () => zoom(-0.2));
      document.getElementById("{zoom_reset_id}")?.addEventListener("click", () => {{
        scale = 1;
        x = 0;
        y = 0;
        update();
      }});

      svg.addEventListener("wheel", (e) => {{
        e.preventDefault();
        zoom(e.deltaY > 0 ? -0.1 : 0.1);
      }}, {{ passive: false }});

      svg.addEventListener("mousedown", (e) => {{
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        svg.style.cursor = "grabbing";
      }});
      window.addEventListener("mouseup", () => {{
        dragging = false;
        svg.style.cursor = "grab";
      }});
      window.addEventListener("mousemove", (e) => {{
        if (!dragging) return;
        x += e.clientX - lastX;
        y += e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        update();
      }});
    }};

    setTimeout(setup, 0);
  }})();
</script>
"""
    components.html(html, height=height + 60, scrolling=True)


def ensure_journey_fields(
    payload: Dict[str, Any],
    intent_name: str,
    scenario_name: str,
    key: str,
) -> Dict[str, Any]:
    if not payload.get("title"):
        payload["title"] = scenario_name
    if not payload.get("description"):
        payload["description"] = f"Journey for {intent_name} - {scenario_name}."
    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        conditions = []
    conditions = [c for c in conditions if isinstance(c, str) and c.strip()]
    if not conditions:
        conditions = [
            f"customer intent is {intent_name}",
            f"customer scenario is {scenario_name}",
        ]
    if not any(f"[key={key}]" in c for c in conditions):
        conditions.append(f"[key={key}]")
    payload["conditions"] = conditions
    return payload


def sanitize_journey_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"fluid", "canned_fluid", "composited_canned", "strict_canned"}
    mode = payload.get("composition_mode")
    if mode and str(mode) not in allowed:
        payload.pop("composition_mode", None)
    if "id" in payload:
        payload.pop("id", None)
    if "tags" in payload:
        payload.pop("tags", None)
    title = payload.get("title")
    if isinstance(title, str) and len(title) > 100:
        payload["title"] = title[:100]
    return payload


def sanitize_guideline_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"fluid", "canned_fluid", "composited_canned", "strict_canned"}
    mode = payload.get("composition_mode")
    if mode and str(mode) not in allowed:
        payload.pop("composition_mode", None)
    if "id" in payload:
        payload.pop("id", None)
    if "tags" in payload:
        payload.pop("tags", None)
    condition = payload.get("condition")
    action = payload.get("action")
    if isinstance(condition, str):
        payload["condition"] = condition.strip()
    if isinstance(action, str):
        payload["action"] = action.strip()
    return payload


def has_required_fields(payload: Dict[str, Any], required: List[str]) -> bool:
    for field in required:
        value = payload.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list) and not value:
            return False
    return True


def sanitize_node_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"fluid", "canned_fluid", "composited_canned", "strict_canned"}
    mode = payload.get("composition_mode")
    if mode and str(mode) not in allowed:
        payload.pop("composition_mode", None)
    if "id" in payload:
        payload.pop("id", None)
    return payload


def sanitize_edge_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "id" in payload:
        payload.pop("id", None)
    return payload


def build_fallback_nodes_edges(
    scenarios: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return [], []


def generate_journey_payload_with_llm(
    client: OpenAI,
    model: str,
    cache: FileCache,
    system_prompt: str,
    user_template: str,
    intent: Dict[str, Any],
    scenario: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    allowed_fields = []
    required_fields = []
    if schema and isinstance(schema, dict):
        props = schema.get("properties", {})
        if isinstance(props, dict):
            allowed_fields = list(props.keys())
        required_fields = schema.get("required", []) or []
    user_prompt = safe_format(
        user_template,
        allowed_fields=allowed_fields or ["title", "description", "conditions", "tags", "id", "name"],
        required_fields=required_fields or ["title", "description", "conditions"],
        key=key,
        intent_json=json.dumps(intent, ensure_ascii=True, indent=2),
        scenario_json=json.dumps(scenario, ensure_ascii=True, indent=2),
    )
    cache_key = sha256_text(f"parlant_journey::{key}::{model}")
    payload = chat_json_with_cache(
        client,
        model,
        system_prompt,
        user_prompt,
        cache=cache,
        cache_key=cache_key,
    )
    if not isinstance(payload, dict):
        return {}
    return payload


def chunk_atoms(items: List[Dict[str, Any]], chunk_size: int = 20) -> List[List[Dict[str, Any]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def generate_cluster_candidates_with_llm(
    llm_client: OpenAI,
    model: str,
    cache: FileCache,
    system_prompt: str,
    user_template: str,
    atoms_rows: List[Dict[str, Any]],
    journey_schema: Optional[Dict[str, Any]],
    guideline_schema: Optional[Dict[str, Any]],
    node_schema: Optional[Dict[str, Any]],
    edge_schema: Optional[Dict[str, Any]],
    chunk_size: int = 20,
) -> Dict[str, Any]:
    journey_fields = []
    journey_required = []
    guideline_fields = []
    guideline_required = []
    node_fields = []
    node_required = []
    edge_fields = []
    edge_required = []
    if journey_schema and isinstance(journey_schema, dict):
        journey_fields = list((journey_schema.get("properties") or {}).keys())
        journey_required = journey_schema.get("required", []) or []
    if guideline_schema and isinstance(guideline_schema, dict):
        guideline_fields = list((guideline_schema.get("properties") or {}).keys())
        guideline_required = guideline_schema.get("required", []) or []
    if node_schema and isinstance(node_schema, dict):
        node_fields = list((node_schema.get("properties") or {}).keys())
        node_required = node_schema.get("required", []) or []
    if edge_schema and isinstance(edge_schema, dict):
        edge_fields = list((edge_schema.get("properties") or {}).keys())
        edge_required = edge_schema.get("required", []) or []

    combined_journeys: List[Dict[str, Any]] = []
    combined_guidelines: List[Dict[str, Any]] = []
    combined_nodes: List[Dict[str, Any]] = []
    combined_edges: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunk_atoms(atoms_rows, chunk_size=chunk_size)):
        user_prompt = safe_format(
            user_template,
            journey_fields=journey_fields or ["title", "description", "conditions", "tags", "id", "name"],
            journey_required=journey_required or ["title", "description", "conditions"],
            guideline_fields=guideline_fields or ["condition", "action"],
            guideline_required=guideline_required or ["condition", "action"],
            node_fields=node_fields or ["action", "description", "tools", "metadata", "composition_mode"],
            node_required=node_required or [],
            edge_fields=edge_fields or ["source", "target", "condition", "metadata"],
            edge_required=edge_required or [],
            atoms_json=json.dumps(chunk, ensure_ascii=True, indent=2),
        )
        cache_key = sha256_text(f"parlant_cluster::{model}::{idx}::{sha256_text(user_prompt)}")
        response = chat_json_with_cache(
            llm_client,
            model,
            system_prompt,
            user_prompt,
            cache=cache,
            cache_key=cache_key,
        )
        if not isinstance(response, dict):
            continue
        journeys = response.get("journeys") or []
        guidelines = response.get("guidelines") or []
        nodes = response.get("journey_nodes") or []
        edges = response.get("journey_edges") or []
        if isinstance(journeys, list):
            combined_journeys.extend([j for j in journeys if isinstance(j, dict)])
        if isinstance(guidelines, list):
            combined_guidelines.extend([g for g in guidelines if isinstance(g, dict)])
        if isinstance(nodes, list):
            combined_nodes.extend([n for n in nodes if isinstance(n, dict)])
        if isinstance(edges, list):
            combined_edges.extend([e for e in edges if isinstance(e, dict)])

    return {
        "journeys": combined_journeys,
        "guidelines": combined_guidelines,
        "journey_nodes": combined_nodes,
        "journey_edges": combined_edges,
    }


def build_journey_index(journeys: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for journey in journeys:
        journey_id = journey.get("id")
        if journey_id:
            index[str(journey_id)] = journey
        tags = journey.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("key:"):
                    index[tag.split("key:", 1)[-1]] = journey
        conditions = journey.get("conditions") or []
        if isinstance(conditions, list):
            for condition in conditions:
                if isinstance(condition, str):
                    match = re.search(r"\[key=([a-f0-9]{64})\]", condition)
                    if match:
                        index[match.group(1)] = journey
        description = journey.get("description") or ""
        if isinstance(description, str):
            match = re.search(r"\[key=([a-f0-9]{64})\]", description)
            if match:
                index[match.group(1)] = journey
    return index


def push_intent_to_parlant(
    intent: Dict[str, Any],
    base_url: str,
    agent_id: Optional[str],
    create_agent_name: Optional[str],
    create_agent_description: Optional[str],
    use_llm: bool,
    llm_client: Optional[OpenAI],
    llm_model: str,
    llm_cache: FileCache,
    llm_system_prompt: str,
    llm_user_template: str,
) -> Dict[str, Any]:
    result = {
        "ok": False,
        "agent_id": None,
        "journeys_created": 0,
        "journeys_skipped": 0,
        "errors": [],
    }
    client = OpenAPIClient(base_url)
    client.load_openapi()

    if create_agent_name:
        agent_payload = {
            "name": create_agent_name.strip(),
            "description": create_agent_description or "",
        }
        try:
            created = client.create_resource("agent", {}, agent_payload)
            agent_id = extract_id(created)
        except OpenAPIError as exc:
            result["errors"].append(str(exc))
            return result
    if not agent_id:
        result["errors"].append("No agent selected or created.")
        return result

    result["agent_id"] = agent_id
    journey_schema = client.get_request_schema("journey", "create")
    try:
        journeys = client.list_resource("journey", {"agent_id": agent_id})
    except OpenAPIError as exc:
        result["errors"].append(str(exc))
        return result

    existing = build_journey_index(journeys)
    scenarios = intent.get("scenarios", []) or []
    for idx, scenario in enumerate(scenarios):
        scenario_id = scenario.get("scenario_id") or scenario.get("id") or f"scenario_{idx}"
        scenario["scenario_id"] = scenario_id
        key = sha256_text(f"{intent.get('intent_id')}|{scenario_id}|{scenario.get('scenario_name')}")
        if key in existing:
            result["journeys_skipped"] += 1
            continue
        if use_llm and llm_client:
            payload = generate_journey_payload_with_llm(
                llm_client,
                llm_model,
                llm_cache,
                llm_system_prompt,
                llm_user_template,
                intent,
                scenario,
                journey_schema,
                key,
            )
        else:
            payload = build_journey_payload(intent, scenario, key, journey_schema)
        payload = ensure_journey_fields(payload, intent.get("intent_name") or "intent", scenario.get("scenario_name") or "scenario", key)
        payload = filter_payload_by_schema(payload, journey_schema)
        payload = sanitize_journey_payload(payload)
        if schema_supports_field(journey_schema, "agent_id"):
            payload["agent_id"] = agent_id
        if schema_supports_field(journey_schema, "agentId"):
            payload["agentId"] = agent_id
        try:
            client.create_resource("journey", {"agent_id": agent_id}, payload)
            result["journeys_created"] += 1
        except OpenAPIError as exc:
            result["errors"].append(str(exc))
    result["ok"] = not result["errors"]
    return result


def build_guideline_index(guidelines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for guideline in guidelines:
        condition = guideline.get("condition") or ""
        action = guideline.get("action") or ""
        text = f"{condition} {action}"
        match = re.search(r"\[key=([a-f0-9]{64})\]", text)
        if match:
            index[match.group(1)] = guideline
    return index


def _best_answer(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    for answer in candidates:
        if not isinstance(answer, str):
            continue
        if answer.strip().upper().startswith("NEEDS_INFO"):
            continue
        if answer.strip():
            return answer.strip()
    return candidates[0].strip() if isinstance(candidates[0], str) and candidates[0].strip() else None


def collect_cluster_scenarios(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scenarios: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in items:
        candidates = row.get("flow_candidates")
        if not candidates and row.get("intent_candidate"):
            candidates = [row]
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            intent_name = (candidate.get("intent_candidate") or "Unknown Intent").strip()
            scenario_name = (candidate.get("scenario_candidate") or "Scenario").strip()
            key = (intent_name, scenario_name)
            if key not in scenarios:
                scenarios[key] = {
                    "intent_id": slugify(intent_name),
                    "intent_name": intent_name,
                    "scenario_id": slugify(scenario_name),
                    "scenario_name": scenario_name,
                    "flow_template": [],
                    "required_fields": [],
                    "handover_rules": [],
                }
            flow_steps = candidate.get("flow_steps_candidate") or []
            if flow_steps and not scenarios[key]["flow_template"]:
                scenarios[key]["flow_template"] = [
                    {"type": "inform", "text": step} for step in flow_steps if isinstance(step, str)
                ]
            resolution = (candidate.get("resolution") or {}).get("summary")
            if resolution and resolution not in scenarios[key].get("handover_rules", []):
                scenarios[key]["handover_rules"].append(resolution)
    return list(scenarios.values())


def build_faq_guidelines(
    items: List[Dict[str, Any]], guideline_schema: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    guidelines: List[Dict[str, Any]] = []
    seen = set()
    for row in items:
        for faq in row.get("faq_candidates", []) or []:
            if not isinstance(faq, dict):
                continue
            question = (faq.get("q_clean") or faq.get("q_raw") or "").strip()
            if not question:
                continue
            answer = _best_answer(faq.get("a_candidates") or [])
            if not answer:
                continue
            dedupe = (question, answer)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            key = sha256_text(f"faq::{question}::{answer}")
            payload = {
                "condition": f"[key={key}] {question}",
                "action": answer,
            }
            payload = filter_payload_by_schema(payload, guideline_schema)
            guidelines.append(payload)
    return guidelines


def build_tone_guidelines(
    items: List[Dict[str, Any]], guideline_schema: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    tones: List[str] = []
    escalations: List[str] = []
    for row in items:
        signals = row.get("blueprint_signals") or {}
        tones.extend([t for t in signals.get("tone", []) if isinstance(t, str)])
        escalations.extend([e for e in signals.get("escalations", []) if isinstance(e, str)])
    guidelines: List[Dict[str, Any]] = []
    if tones:
        tone_text = ", ".join(list(dict.fromkeys(tones))[:4])
        key = sha256_text(f"tone::{tone_text}")
        payload = {
            "condition": f"[key={key}] conversation requires tone guidance",
            "action": f"Use a tone that is: {tone_text}.",
        }
        guidelines.append(filter_payload_by_schema(payload, guideline_schema))
    if escalations:
        unique = list(dict.fromkeys(escalations))[:5]
        key = sha256_text("escalations::" + "|".join(unique))
        payload = {
            "condition": f"[key={key}] escalation signals detected",
            "action": "Escalate to a human when you detect: " + "; ".join(unique) + ".",
        }
        guidelines.append(filter_payload_by_schema(payload, guideline_schema))
    return guidelines


def preview_cluster_candidates(
    atoms_rows: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    base_url: str,
    use_llm: bool,
    llm_client: Optional[OpenAI],
    llm_model: str,
    llm_cache: FileCache,
    llm_system_prompt: str,
    llm_user_template: str,
) -> Dict[str, Any]:
    result = {"ok": False, "journeys": [], "guidelines": [], "errors": []}
    client = OpenAPIClient(base_url)
    client.load_openapi()
    journey_schema = client.get_request_schema("journey", "create")
    guideline_schema = None
    node_schema = None
    edge_schema = None
    try:
        guideline_schema = client.get_request_schema("guideline", "create")
    except OpenAPIError:
        guideline_schema = None
    try:
        node_schema = client.get_request_schema("node", "create")
    except OpenAPIError:
        node_schema = None
    try:
        edge_schema = client.get_request_schema("edge", "create")
    except OpenAPIError:
        edge_schema = None
    journeys = []
    guidelines = []
    nodes_preview: List[Dict[str, Any]] = []
    edges_preview: List[Dict[str, Any]] = []
    if use_llm and llm_client:
        llm_result = generate_cluster_candidates_with_llm(
            llm_client,
            llm_model,
            llm_cache,
            llm_system_prompt,
            llm_user_template,
            atoms_rows,
            journey_schema,
            guideline_schema,
            node_schema,
            edge_schema,
        )
        journey_required = journey_schema.get("required", []) if isinstance(journey_schema, dict) else []
        guideline_required = guideline_schema.get("required", []) if isinstance(guideline_schema, dict) else []
        node_required = node_schema.get("required", []) if isinstance(node_schema, dict) else []
        edge_required = edge_schema.get("required", []) if isinstance(edge_schema, dict) else []
        for payload in llm_result.get("journeys", []):
            payload = filter_payload_by_schema(payload, journey_schema)
            title = payload.get("title") or payload.get("name") or "journey"
            conditions = payload.get("conditions") or []
            key = sha256_text(f"{title}|{json.dumps(conditions, ensure_ascii=True)}")
            payload = ensure_journey_fields(payload, title, title, key)
            payload = sanitize_journey_payload(payload)
            if journey_required and not has_required_fields(payload, journey_required):
                continue
            journeys.append({"scenario_name": title, "payload": payload, "key": key})
        for payload in llm_result.get("guidelines", []):
            payload = filter_payload_by_schema(payload, guideline_schema)
            condition = payload.get("condition", "")
            action = payload.get("action", "")
            key = sha256_text(f"{condition}|{action}")
            if "[key=" not in str(condition):
                payload["condition"] = f"[key={key}] {condition}".strip()
            payload = sanitize_guideline_payload(payload)
            if guideline_required and not has_required_fields(payload, guideline_required):
                continue
            guidelines.append(payload)
        for raw in llm_result.get("journey_nodes", []):
            if not isinstance(raw, dict):
                continue
            journey_title = raw.get("journey_title") or raw.get("journey") or raw.get("title")
            label = raw.get("label") or raw.get("node_label") or raw.get("name")
            payload = filter_payload_by_schema(raw, node_schema)
            payload = sanitize_node_payload(payload)
            if node_required and not has_required_fields(payload, node_required):
                continue
            nodes_preview.append(
                {
                    "journey_title": journey_title,
                    "label": label,
                    "payload": payload,
                }
            )
        for raw in llm_result.get("journey_edges", []):
            if not isinstance(raw, dict):
                continue
            journey_title = raw.get("journey_title") or raw.get("journey") or raw.get("title")
            source_label = raw.get("source_label") or raw.get("source") or raw.get("from")
            target_label = raw.get("target_label") or raw.get("target") or raw.get("to")
            payload = filter_payload_by_schema(raw, edge_schema)
            payload = sanitize_edge_payload(payload)
            if edge_required and not has_required_fields(payload, edge_required):
                continue
            edges_preview.append(
                {
                    "journey_title": journey_title,
                    "source_label": source_label,
                    "target_label": target_label,
                    "payload": payload,
                }
            )
        if not nodes_preview or not edges_preview:
            result["errors"].append(
                "LLM did not return journey_nodes and journey_edges. Strict mode enabled."
            )
            result["ok"] = False
    else:
        scenarios = collect_cluster_scenarios(items)
        for scenario in scenarios:
            intent = {
                "intent_id": scenario["intent_id"],
                "intent_name": scenario["intent_name"],
                "definition": "",
            }
            scenario_id = scenario.get("scenario_id") or slugify(scenario.get("scenario_name", "scenario"))
            key = sha256_text(f"{intent['intent_id']}|{scenario_id}|{scenario.get('scenario_name')}")
            payload = build_journey_payload(intent, scenario, key, journey_schema)
            payload = ensure_journey_fields(payload, intent["intent_name"], scenario["scenario_name"], key)
            payload = filter_payload_by_schema(payload, journey_schema)
            journeys.append({"scenario_name": scenario["scenario_name"], "payload": payload, "key": key})
        if guideline_schema is not None:
            guidelines.extend(build_faq_guidelines(items, guideline_schema))
            guidelines.extend(build_tone_guidelines(items, guideline_schema))

    result["journeys"] = journeys
    result["guidelines"] = guidelines
    result["journey_nodes"] = nodes_preview
    result["journey_edges"] = edges_preview
    result["ok"] = True
    return result


def push_cluster_candidates(
    atoms_rows: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    base_url: str,
    agent_id: Optional[str],
    create_agent_name: Optional[str],
    create_agent_description: Optional[str],
    use_llm: bool,
    llm_client: Optional[OpenAI],
    llm_model: str,
    llm_cache: FileCache,
    llm_system_prompt: str,
    llm_user_template: str,
) -> Dict[str, Any]:
    result = {
        "ok": False,
        "agent_id": None,
        "journeys_created": 0,
        "journeys_skipped": 0,
        "guidelines_created": 0,
        "guidelines_skipped": 0,
        "errors": [],
    }
    client = OpenAPIClient(base_url)
    client.load_openapi()

    if create_agent_name:
        agent_payload = {
            "name": create_agent_name.strip(),
            "description": create_agent_description or "",
        }
        try:
            created = client.create_resource("agent", {}, agent_payload)
            agent_id = extract_id(created)
        except OpenAPIError as exc:
            result["errors"].append(str(exc))
            return result
    if not agent_id:
        result["errors"].append("No agent selected or created.")
        return result

    result["agent_id"] = agent_id
    preview = preview_cluster_candidates(
        atoms_rows,
        items,
        base_url,
        use_llm,
        llm_client,
        llm_model,
        llm_cache,
        llm_system_prompt,
        llm_user_template,
    )
    if not preview.get("ok"):
        result["errors"].extend(preview.get("errors") or [])
        return result

    journeys = preview.get("journeys") or []
    guidelines = preview.get("guidelines") or []
    journey_nodes = preview.get("journey_nodes") or []
    journey_edges = preview.get("journey_edges") or []

    existing_journeys = client.list_resource("journey", {"agent_id": agent_id})
    journey_index = build_journey_index(existing_journeys)
    journey_title_map = {
        normalize_title(str(j.get("title"))): extract_id(j)
        for j in existing_journeys
        if j.get("title") and extract_id(j)
    }
    journey_schema = client.get_request_schema("journey", "create")
    journey_required = journey_schema.get("required", []) if isinstance(journey_schema, dict) else []
    for entry in journeys:
        payload = entry.get("payload") or {}
        key = entry.get("key")
        if key and key in journey_index:
            result["journeys_skipped"] += 1
            continue
        payload = sanitize_journey_payload(payload)
        if journey_required and not has_required_fields(payload, journey_required):
            result["errors"].append("Journey payload missing required fields; skipped.")
            continue
        if schema_supports_field(client.get_request_schema("journey", "create"), "agent_id"):
            payload["agent_id"] = agent_id
        if schema_supports_field(client.get_request_schema("journey", "create"), "agentId"):
            payload["agentId"] = agent_id
        try:
            created = client.create_resource("journey", {"agent_id": agent_id}, payload)
            result["journeys_created"] += 1
            journey_id = extract_id(created)
            if journey_id:
                journey_title_map[normalize_title(str(payload.get("title")))] = journey_id
        except OpenAPIError as exc:
            result["errors"].append(str(exc))

    try:
        existing_guidelines = client.list_resource("guideline", {"agent_id": agent_id})
    except OpenAPIError:
        existing_guidelines = []
    guideline_schema = None
    try:
        guideline_schema = client.get_request_schema("guideline", "create")
    except OpenAPIError:
        guideline_schema = None
    guideline_required = guideline_schema.get("required", []) if isinstance(guideline_schema, dict) else []
    guideline_index = build_guideline_index(existing_guidelines)
    for payload in guidelines:
        payload = sanitize_guideline_payload(payload)
        if guideline_required and not has_required_fields(payload, guideline_required):
            result["errors"].append("Guideline payload missing required fields; skipped.")
            continue
        condition = payload.get("condition", "")
        match = re.search(r"\[key=([a-f0-9]{64})\]", str(condition))
        if match and match.group(1) in guideline_index:
            result["guidelines_skipped"] += 1
            continue
        try:
            client.create_resource("guideline", {"agent_id": agent_id}, payload)
            result["guidelines_created"] += 1
        except OpenAPIError as exc:
            result["errors"].append(str(exc))

    if journey_nodes or journey_edges:
        for journey_title, journey_id in journey_title_map.items():
            if not journey_id:
                continue
            try:
                existing_nodes = client.list_resource("node", {"journey_id": journey_id})
            except OpenAPIError:
                existing_nodes = []
            label_map = {}
            root_id = None
            for node in existing_nodes:
                if node.get("id") and node.get("action") in (None, "", "null"):
                    root_id = extract_id(node)
                metadata = node.get("metadata") or {}
                label = None
                if isinstance(metadata, dict):
                    label = metadata.get("label") or (metadata.get("journey_node") or {}).get("label")
                if label:
                    label_map[str(label)] = extract_id(node)
            if root_id:
                label_map["ROOT"] = root_id

            for entry in journey_nodes:
                if normalize_title(str(entry.get("journey_title") or "")) != str(journey_title):
                    continue
                label = entry.get("label")
                payload = entry.get("payload") or {}
                if not label or not isinstance(payload, dict):
                    continue
                if label in label_map:
                    continue
                metadata = payload.get("metadata") or {}
                if isinstance(metadata, dict):
                    metadata.setdefault("journey_node", {})
                    if isinstance(metadata.get("journey_node"), dict):
                        metadata["journey_node"].setdefault("label", label)
                        metadata["journey_node"].setdefault("kind", "state")
                    metadata.setdefault("label", label)
                payload["metadata"] = metadata
                try:
                    created = client.create_resource("node", {"journey_id": journey_id}, payload)
                    node_id = extract_id(created)
                    if node_id:
                        label_map[str(label)] = node_id
                except OpenAPIError as exc:
                    result["errors"].append(str(exc))

            try:
                existing_edges = client.list_resource("edge", {"journey_id": journey_id})
            except OpenAPIError:
                existing_edges = []
            edge_index = {(e.get("source"), e.get("target"), e.get("condition")) for e in existing_edges}
            for entry in journey_edges:
                if normalize_title(str(entry.get("journey_title") or "")) != str(journey_title):
                    continue
                source_label = entry.get("source_label")
                target_label = entry.get("target_label")
                payload = entry.get("payload") or {}
                if not source_label or not target_label or not isinstance(payload, dict):
                    continue
                if str(source_label) == "ROOT" and label_map.get("ROOT"):
                    source_id = label_map.get("ROOT")
                else:
                    source_id = label_map.get(str(source_label))
                target_id = label_map.get(str(target_label))
                if not source_id or not target_id:
                    continue
                payload["source"] = source_id
                payload["target"] = target_id
                signature = (payload.get("source"), payload.get("target"), payload.get("condition"))
                if signature in edge_index:
                    continue
                try:
                    client.create_resource("edge", {"journey_id": journey_id}, payload)
                except OpenAPIError as exc:
                    result["errors"].append(str(exc))

    result["ok"] = not result["errors"]
    return result


def preview_intent_journeys(
    intent: Dict[str, Any],
    base_url: str,
    use_llm: bool,
    llm_client: Optional[OpenAI],
    llm_model: str,
    llm_cache: FileCache,
    llm_system_prompt: str,
    llm_user_template: str,
) -> Dict[str, Any]:
    result = {"ok": False, "journeys": [], "errors": []}
    client = OpenAPIClient(base_url)
    client.load_openapi()
    journey_schema = client.get_request_schema("journey", "create")
    scenarios = intent.get("scenarios", []) or []
    for idx, scenario in enumerate(scenarios):
        scenario_id = scenario.get("scenario_id") or scenario.get("id") or f"scenario_{idx}"
        scenario["scenario_id"] = scenario_id
        key = sha256_text(f"{intent.get('intent_id')}|{scenario_id}|{scenario.get('scenario_name')}")
        if use_llm and llm_client:
            payload = generate_journey_payload_with_llm(
                llm_client,
                llm_model,
                llm_cache,
                llm_system_prompt,
                llm_user_template,
                intent,
                scenario,
                journey_schema,
                key,
            )
        else:
            payload = build_journey_payload(intent, scenario, key, journey_schema)
        payload = ensure_journey_fields(
            payload,
            intent.get("intent_name") or "intent",
            scenario.get("scenario_name") or "scenario",
            key,
        )
        payload = filter_payload_by_schema(payload, journey_schema)
        result["journeys"].append(
            {
                "scenario_name": scenario.get("scenario_name") or scenario_id,
                "payload": payload,
            }
        )
    result["ok"] = True
    return result


def read_cache_stats(log_path: Path) -> Dict[str, int]:
    stats = {"cache_hit": 0, "cache_miss": 0}
    if not log_path.exists():
        return stats
    for line in log_path.read_text().splitlines():
        if "cache_hit" in line:
            stats["cache_hit"] += 1
        if "cache_miss" in line:
            stats["cache_miss"] += 1
    return stats


def save_run_state(ws_root: Path, state: Dict[str, Any]) -> None:
    save_json(ws_root / "run_state.json", state)


def load_run_state(ws_root: Path) -> Dict[str, Any]:
    path = ws_root / "run_state.json"
    if path.exists():
        return load_json(path)
    return {}


def mark_stage(state: Dict[str, Any], stage: str) -> None:
    state[stage] = datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_ui_state() -> Dict[str, Any]:
    path = WORKSPACE_ROOT / ".ui_state.json"
    if path.exists():
        try:
            return load_json(path)
        except json.JSONDecodeError:
            return {}
    return {}


def save_ui_state(state: Dict[str, Any]) -> None:
    save_json(WORKSPACE_ROOT / ".ui_state.json", state)


def load_prompts(ws_root: Path) -> Dict[str, str]:
    path = ws_root / "prompts.json"
    if path.exists():
        return load_json(path)
    return {}


def save_prompts(ws_root: Path, prompts: Dict[str, str]) -> None:
    save_json(ws_root / "prompts.json", prompts)


def load_parlant_cluster_cache(ws_root: Path) -> Dict[str, Any]:
    path = ws_root / "parlant_cluster_cache.json"
    if path.exists():
        try:
            return load_json(path)
        except json.JSONDecodeError:
            return {}
    return {}


def save_parlant_cluster_cache(ws_root: Path, payload: Dict[str, Any]) -> None:
    save_json(ws_root / "parlant_cluster_cache.json", payload)


def get_prompt(prompts: Dict[str, str], key: str, default: str) -> str:
    value = prompts.get(key)
    return value if isinstance(value, str) and value.strip() else default


def edit_prompt_dialog(
    ws_root: Path,
    prompts: Dict[str, str],
    key_prefix: str,
    title: str,
    system_default: str,
    user_default: str,
) -> None:
    @st.dialog(title)
    def _dialog() -> None:
        st.caption("Edit the system and user prompts. Keep required placeholders intact.")
        system_key = f"{key_prefix}.system"
        user_key = f"{key_prefix}.user"
        system_text = st.text_area(
            "System prompt",
            value=get_prompt(prompts, system_key, system_default),
            height=180,
        )
        user_text = st.text_area(
            "User prompt",
            value=get_prompt(prompts, user_key, user_default),
            height=320,
        )
        if st.button("Save prompt"):
            prompts[system_key] = system_text
            prompts[user_key] = user_text
            save_prompts(ws_root, prompts)
            st.success("Saved prompt.")

    _dialog()


def edit_single_prompt_dialog(
    ws_root: Path,
    prompts: Dict[str, str],
    key: str,
    title: str,
    default_text: str,
) -> None:
    @st.dialog(title)
    def _dialog() -> None:
        st.caption("Edit the prompt. Keep required placeholders (e.g., {intent_json}) intact.")
        text = st.text_area(
            "Prompt",
            value=get_prompt(prompts, key, default_text),
            height=320,
        )
        if st.button("Save prompt"):
            prompts[key] = text
            save_prompts(ws_root, prompts)
            st.success("Saved prompt.")

    _dialog()


def normalize_intents(intents: List[Dict[str, Any]]) -> tuple[list[dict], bool]:
    legacy = False
    normalized = []
    for intent in intents:
        item = dict(intent)
        if "intent_confidence" not in item and "confidence" in item:
            item["intent_confidence"] = item.get("confidence")
            legacy = True
        scenarios = item.get("scenarios", []) or []
        updated_scenarios = []
        for scenario in scenarios:
            scenario_item = dict(scenario)
            if "scenario_confidence" not in scenario_item:
                scenario_item["scenario_confidence"] = None
                legacy = True
            if "automation_suitability" not in scenario_item:
                scenario_item["automation_suitability"] = None
                legacy = True
            updated_scenarios.append(scenario_item)
        item["scenarios"] = updated_scenarios
        normalized.append(item)
    return normalized, legacy


def _call_chat_messages(
    client: OpenAI,
    model: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    payload = [{"role": "system", "content": system_prompt}] + messages
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "temperature" not in msg or ("unsupported" not in msg and "does not support" not in msg):
            raise
        resp = client.chat.completions.create(
            model=model,
            messages=payload,
        )
    return resp.choices[0].message.content or ""


def _format_transcript(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role") or "other"
        text = msg.get("text") or msg.get("content") or ""
        lines.append(f"{role}: {text}".strip())
    return "\n".join(lines).strip()


def _pick_cluster_for_intent(
    flow_items: List[Dict[str, Any]],
    clusters: List[List[int]],
    intent: Dict[str, Any],
) -> int | None:
    index_to_cluster = {}
    for idx, cluster in enumerate(clusters):
        for item_idx in cluster:
            index_to_cluster[item_idx] = idx

    example_ids: List[str] = []
    for scenario in intent.get("scenarios") or []:
        for cid in scenario.get("example_conversation_ids") or []:
            if cid:
                example_ids.append(str(cid))
    example_ids = list(dict.fromkeys(example_ids))

    counts: Dict[int, int] = {}
    if example_ids:
        for idx, item in enumerate(flow_items):
            convo_id = str(item.get("conversation_id") or "")
            if convo_id and convo_id in example_ids:
                cluster_idx = index_to_cluster.get(idx)
                if cluster_idx is not None:
                    counts[cluster_idx] = counts.get(cluster_idx, 0) + 1

    if not counts:
        intent_name = (intent.get("intent_name") or "").strip().lower()
        scenario_names = [
            (s.get("scenario_name") or "").strip().lower()
            for s in (intent.get("scenarios") or [])
        ]
        for idx, item in enumerate(flow_items):
            cand_intent = (item.get("intent_candidate") or "").strip().lower()
            cand_scenario = (item.get("scenario_candidate") or "").strip().lower()
            match = False
            if intent_name and (intent_name in cand_intent or cand_intent in intent_name):
                match = True
            if not match and scenario_names:
                for name in scenario_names:
                    if name and (name in cand_scenario or cand_scenario in name):
                        match = True
                        break
            if match:
                cluster_idx = index_to_cluster.get(idx)
                if cluster_idx is not None:
                    counts[cluster_idx] = counts.get(cluster_idx, 0) + 1

    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]
    if clusters:
        return 0
    return None


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        rmtree(path)
    else:
        path.unlink()


def list_artifact_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    files = [p for p in path.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


st.title("Conversation-Driven Intent/FAQ/Blueprint Extraction")

notice = st.session_state.pop("intent_update_notice", None)
if notice:
    st.success(notice)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spline+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Spline Sans', sans-serif;
}

.faq-card, .intent-card {
  border: 1px solid #e6e1d6;
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 12px;
  background: linear-gradient(180deg, #fffdf7, #fff9ef);
  box-shadow: 0 8px 18px rgba(31, 27, 16, 0.06);
}

.faq-title {
  font-size: 18px;
  font-weight: 700;
  color: #1e1b12;
}

.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-right: 6px;
}

.pill-high { background: #1f4e3d; color: #f1f7f2; }
.pill-mid { background: #d9980f; color: #fff8e1; }
.pill-low { background: #b3472b; color: #fff1ed; }
.pill-neutral { background: #ece6da; color: #4b3f2f; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
}
.badge-review { background: #ffe7e0; color: #8a2c16; }
.badge-ok { background: #e5f6ea; color: #1a6b34; }

.flow-card {
  border: 1px solid #ded7c8;
  border-radius: 18px;
  padding: 18px;
  background: #fefaf1;
  margin: 10px 0 16px 0;
}

.flowchart-frame {
  border: 2px solid #cdb892;
  border-radius: 16px;
  padding: 10px;
  background: #fff7e4;
  margin: 10px 0 16px 0;
  box-shadow: inset 0 0 0 1px rgba(125, 98, 53, 0.12);
}

iframe[title="streamlit_agraph.agraph"],
iframe[title^="streamlit_agraph"],
iframe[src*="streamlit_agraph"] {
  border: 2px solid #cdb892 !important;
  border-radius: 16px !important;
  background: #fff7e4 !important;
}

.flow-step {
  border: 1px solid #efe4cc;
  border-radius: 12px;
  padding: 10px 12px;
  background: #fffdf8;
  margin: 8px 0;
}

.flow-step .type {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #7a6437;
  text-transform: uppercase;
}

.flow-arrow {
  text-align: center;
  color: #b79a5d;
  font-size: 18px;
  margin: 4px 0;
}

.detail-shell {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.detail-card {
  border: 1px solid #e3d8c5;
  border-radius: 18px;
  padding: 16px 18px;
  background: linear-gradient(135deg, #fffaf1, #fff6e3);
  box-shadow: 0 10px 22px rgba(34, 28, 15, 0.08);
}

.detail-title {
  font-size: 20px;
  font-weight: 700;
  color: #1f1b12;
}

.detail-subtitle {
  font-size: 13px;
  color: #6b5b43;
  margin-top: 6px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: #f2ead7;
  color: #4a3b26;
}

.muted {
  color: #7a6a52;
  font-size: 13px;
}

.section-title {
  font-size: 14px;
  font-weight: 700;
  color: #3a2f1e;
  margin: 8px 0 6px 0;
}

.intent-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 12px;
  background: #fff7e4;
  border: 1px solid #eadfcb;
  margin-bottom: 8px;
}

.intent-name {
  font-size: 13px;
  font-weight: 600;
  color: #2f2618;
}
</style>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Project")
    ui_state = load_ui_state()
    existing = list_projects()
    if "project_select" not in st.session_state:
        last_project = ui_state.get("last_project_id", "")
        st.session_state["project_select"] = last_project if last_project in existing else ""
    if "project_new" not in st.session_state:
        st.session_state["project_new"] = ui_state.get("last_project_new", "indent_discovery_v2")
    selected = st.selectbox(
        "Select project",
        options=[""] + existing,
        format_func=lambda x: x or "Choose...",
        key="project_select",
    )
    new_project = st.text_input("Or create new project id", key="project_new")
    project_id = selected or new_project
    st.write(f"Active project: **{project_id}**")

    ws = get_workspace(WORKSPACE_ROOT, project_id)
    run_state = load_run_state(ws.root)
    prompts = load_prompts(ws.root)
    parlant_cluster_cache = load_parlant_cluster_cache(ws.root)

    st.divider()
    st.header("Upload")
    upload_file = st.file_uploader(
        "Upload conversations (.json, .jsonl, or .txt)", type=["json", "jsonl", "txt"]
    )
    if upload_file is not None:
        raw_path = ws.input_dir / upload_file.name
        ws.input_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(upload_file.getvalue())
        st.success(f"Saved to {raw_path.name}")

    st.divider()
    st.header("Settings")
    project_settings = (ui_state.get("projects") or {}).get(project_id, {})
    if st.session_state.get("active_project") != project_id:
        st.session_state["active_project"] = project_id
        st.session_state["llm_model"] = project_settings.get("llm_model", "gpt-4o-mini")
        st.session_state["embed_model"] = project_settings.get("embed_model", "text-embedding-3-small")
        st.session_state["threshold"] = project_settings.get("threshold", 0.82)
        st.session_state["min_faq_freq"] = project_settings.get("min_faq_freq", 3)
        st.session_state["min_intent_freq"] = project_settings.get("min_intent_freq", 3)
        st.session_state["min_intent_cluster_size"] = project_settings.get("min_intent_cluster_size", 3)
        st.session_state["max_candidates_per_cluster"] = project_settings.get("max_candidates_per_cluster", 25)
        st.session_state["cluster_on_intent_only"] = project_settings.get("cluster_on_intent_only", True)
        st.session_state["redact_pii"] = project_settings.get("redact_pii", False)
        st.session_state["openai_key"] = project_settings.get("openai_key", "") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        st.session_state["blueprint_use_llm"] = project_settings.get("blueprint_use_llm", False)
        st.session_state["simulation_model"] = project_settings.get(
            "simulation_model", project_settings.get("llm_model", "gpt-4o-mini")
        )
        st.session_state["judge_model"] = project_settings.get(
            "judge_model", project_settings.get("llm_model", "gpt-4o-mini")
        )
        st.session_state["parlant_enabled"] = project_settings.get(
            "parlant_enabled", os.environ.get("PARLANT_ENABLED", "").lower() == "true"
        )
        st.session_state["parlant_base_url"] = project_settings.get(
            "parlant_base_url", os.environ.get("PARLANT_BASE_URL", "http://localhost:8800")
        )

    llm_model = st.text_input("LLM model", key="llm_model")
    embed_model = st.text_input("Embedding model", key="embed_model")
    threshold = st.slider("Similarity threshold", 0.5, 0.95, 0.82, 0.01, key="threshold")
    min_faq_freq = st.number_input("Min FAQ frequency", min_value=1, value=3, step=1, key="min_faq_freq")
    min_intent_freq = st.number_input("Min intent frequency", min_value=1, value=3, step=1, key="min_intent_freq")
    min_intent_cluster_size = st.number_input(
        "Min intent cluster size",
        min_value=1,
        value=3,
        step=1,
        help="Only generate intents from clusters with size >= this value.",
        key="min_intent_cluster_size",
    )
    max_candidates_per_cluster = st.number_input(
        "Max candidates per cluster (batch size)",
        min_value=5,
        value=25,
        step=5,
        help="Clusters larger than this will be split into batches for LLM processing.",
        key="max_candidates_per_cluster",
    )
    cluster_on_intent_only = st.checkbox("Cluster on intent only (ignore scenario)", key="cluster_on_intent_only")
    redact_pii = st.checkbox("Redact PII", key="redact_pii")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", key="openai_key")
    blueprint_use_llm = st.checkbox("Use LLM for blueprint", key="blueprint_use_llm")

    st.divider()
    st.header("Parlant")
    parlant_enabled = st.checkbox("Enable Parlant", key="parlant_enabled")
    parlant_base_url = st.text_input(
        "PARLANT_BASE_URL",
        key="parlant_base_url",
        help="Base URL for the Parlant server.",
    )
    if st.button("Edit Parlant journey prompt", key="parlant_edit_journey_prompt"):
        edit_prompt_dialog(
            ws.root,
            prompts,
            "parlant_journey",
            "Edit Parlant Journey Prompt",
            PARLANT_JOURNEY_SYSTEM_PROMPT,
            PARLANT_JOURNEY_USER_TEMPLATE,
        )
    if st.button("Edit Parlant cluster prompt", key="parlant_edit_cluster_prompt"):
        edit_prompt_dialog(
            ws.root,
            prompts,
            "parlant_cluster",
            "Edit Parlant Cluster Prompt",
            PARLANT_CLUSTER_SYSTEM_PROMPT,
            PARLANT_CLUSTER_USER_TEMPLATE,
        )
    test_parlant = st.button("Test connection", key="parlant_test_connection")
    if test_parlant:
        result = test_parlant_connection(parlant_base_url)
        st.session_state["parlant_connection_ok"] = result.get("ok", False)
        st.session_state["parlant_connection_msg"] = result.get("error", "")
        st.session_state["parlant_last_check_ts"] = time.time()
        if result.get("ok"):
            st.success("Parlant connection OK.")
            warning = result.get("warning")
            if warning:
                st.warning(warning)
        else:
            msg = result.get("error") or "Parlant connection failed."
            st.error(msg)
    if st.button("Show server start steps", key="parlant_show_steps"):
        st.markdown("**macOS / Linux**")
        st.code(
            "pip install parlant\n"
            "export EMCIE_API_KEY=\"your_key_here\"\n"
            "python parlant_server/main.py",
            language="bash",
        )
        st.markdown("**macOS / Linux (OpenAI mode)**")
        st.code(
            "pip install parlant\n"
            "export OPENAI_API_KEY=\"your_key_here\"\n"
            "python parlant_server/main.py --openai",
            language="bash",
        )
        st.markdown("**Windows PowerShell**")
        st.code(
            "pip install parlant\n"
            "$env:EMCIE_API_KEY=\"your_key_here\"\n"
            "python parlant_server\\main.py",
            language="powershell",
        )
        st.markdown("**Windows PowerShell (OpenAI mode)**")
        st.code(
            "pip install parlant\n"
            "$env:OPENAI_API_KEY=\"your_key_here\"\n"
            "python parlant_server\\main.py --openai",
            language="powershell",
        )
    if st.button("Sync journeys now", key="parlant_sync_now"):
        if not parlant_enabled:
            st.warning("Enable Parlant before syncing.")
        else:
            sync_result = sync_project_to_parlant(ws.root, parlant_base_url)
            st.session_state["parlant_last_sync"] = sync_result
            if sync_result.get("ok"):
                if not sync_result.get("skipped"):
                    mark_stage(run_state, "parlant_sync")
                    save_run_state(ws.root, run_state)
                if sync_result.get("skipped"):
                    st.info("Parlant is already in sync.")
                else:
                    st.success("Parlant sync complete.")
            else:
                st.error("Parlant sync failed. Check logs for details.")

    ui_state["last_project_id"] = project_id
    ui_state["last_project_new"] = new_project
    ui_state.setdefault("projects", {})
    ui_state["projects"][project_id] = {
        "llm_model": llm_model,
        "embed_model": embed_model,
        "threshold": threshold,
        "min_faq_freq": min_faq_freq,
        "min_intent_freq": min_intent_freq,
        "min_intent_cluster_size": min_intent_cluster_size,
        "max_candidates_per_cluster": max_candidates_per_cluster,
        "cluster_on_intent_only": cluster_on_intent_only,
        "redact_pii": redact_pii,
        "openai_key": openai_key,
        "blueprint_use_llm": blueprint_use_llm,
        "simulation_model": st.session_state.get("simulation_model", llm_model),
        "judge_model": st.session_state.get("judge_model", llm_model),
        "parlant_enabled": parlant_enabled,
        "parlant_base_url": parlant_base_url,
    }
    save_ui_state(ui_state)

    llm_cache = FileCache(ws.llm_cache_dir, ws.logs_dir / "cache_llm.log")
    embed_cache = FileCache(ws.embed_cache_dir, ws.logs_dir / "cache_embed.log")

    st.divider()
    st.header("Run stages")
    run_all = st.button("Run All", type="primary")
    run_normalize = st.button("Normalize conversations")

    atoms_cols = st.columns([2, 1])
    with atoms_cols[0]:
        run_atoms = st.button("Extract atoms")
    with atoms_cols[1]:
        edit_atoms = st.button("Edit prompt", key="edit_atoms_prompt")

    faqs_cols = st.columns([2, 1])
    with faqs_cols[0]:
        run_faqs = st.button("Build FAQs")
    with faqs_cols[1]:
        edit_faqs = st.button("Edit prompt", key="edit_faqs_prompt")

    intents_cols = st.columns([2, 1])
    with intents_cols[0]:
        run_intents = st.button("Build Intents/Flows")
    with intents_cols[1]:
        edit_intents = st.button("Edit prompt", key="edit_intents_prompt")

    blueprint_cols = st.columns([2, 1])
    with blueprint_cols[0]:
        run_blueprint = st.button("Build Blueprint")
    with blueprint_cols[1]:
        edit_blueprint = st.button("Edit prompt", key="edit_blueprint_prompt")
    publish_parlant = st.button("Publish to Parlant")
    st.divider()
    st.subheader("Clear outputs")
    clear_all_outputs = st.button("Clear all outputs")
    clear_normalize = st.button("Clear normalization output")
    clear_atoms = st.button("Clear atoms output")
    clear_faqs = st.button("Clear FAQs output")
    clear_intents = st.button("Clear intents/flows output")
    clear_blueprint = st.button("Clear blueprint output")
    st.divider()
    st.subheader("Embeddings")
    clear_embeddings = st.button("Clear embeddings cache")
    precompute_embeddings = st.button("Generate embeddings")

    st.divider()
    st.header("Cache stats")
    llm_stats = read_cache_stats(ws.logs_dir / "cache_llm.log")
    embed_stats = read_cache_stats(ws.logs_dir / "cache_embed.log")
    st.write(f"LLM cache hits: {llm_stats['cache_hit']} | misses: {llm_stats['cache_miss']}")
    st.write(f"Embed cache hits: {embed_stats['cache_hit']} | misses: {embed_stats['cache_miss']}")

    if run_state:
        st.divider()
        st.header("Last run")
        for stage, ts in run_state.items():
            st.write(f"{stage}: {ts}")


client = get_openai_client(openai_key)


def build_flow_items(atoms_rows: List[Dict[str, Any]], intent_only: bool) -> List[Dict[str, Any]]:
    flow_items = []
    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id")
        for candidate in atoms.get("flow_candidates", []) or []:
            intent = (candidate.get("intent_candidate") or "").strip()
            scenario = (candidate.get("scenario_candidate") or "").strip()
            if not intent and not scenario:
                continue
            flow_items.append(
                {
                    "conversation_id": convo_id,
                    "intent_candidate": intent,
                    "scenario_candidate": scenario,
                    "flow_steps_candidate": candidate.get("flow_steps_candidate") or [],
                    "resolution": candidate.get("resolution") or {},
                    "faq_candidates": atoms.get("faq_candidates") or [],
                    "blueprint_signals": atoms.get("blueprint_signals") or {},
                    "flow_key": intent if intent_only else f"{intent} | {scenario}".strip(" |"),
                }
            )
    return flow_items

if edit_atoms:
    edit_prompt_dialog(
        ws.root,
        prompts,
        "atoms",
        "Edit Atoms Prompt",
        ATOMS_SYSTEM_PROMPT,
        ATOMS_USER_TEMPLATE,
    )

if edit_faqs:
    edit_prompt_dialog(
        ws.root,
        prompts,
        "faq",
        "Edit FAQ Prompt",
        FAQ_SYSTEM_PROMPT,
        FAQ_USER_TEMPLATE,
    )

if edit_intents:
    edit_prompt_dialog(
        ws.root,
        prompts,
        "intent_flow",
        "Edit Intent/Flow Prompt",
        INTENT_SYSTEM_PROMPT,
        INTENT_USER_TEMPLATE,
    )

if edit_blueprint:
    edit_prompt_dialog(
        ws.root,
        prompts,
        "blueprint",
        "Edit Blueprint Prompt",
        BLUEPRINT_SYSTEM_PROMPT,
        BLUEPRINT_USER_TEMPLATE,
    )

if clear_all_outputs:
    _remove_path(ws.normalized_path)
    _remove_path(ws.atoms_path)
    _remove_path(ws.faq_catalogue_path)
    _remove_path(ws.intent_flow_catalogue_path)
    _remove_path(ws.blueprint_path)
    _remove_path(ws.root / "logs")
    _remove_path(ws.root / "run_state.json")
    st.success("Cleared all outputs and run state.")

if clear_embeddings:
    _remove_path(ws.embed_cache_dir)
    st.success("Cleared embeddings cache.")

if precompute_embeddings:
    if not ws.atoms_path.exists():
        st.warning("Run atoms extraction first.")
    else:
        atoms_rows = load_jsonl(ws.atoms_path)
        flow_keys = []
        faq_questions = []
        for atoms in atoms_rows:
            for candidate in atoms.get("flow_candidates", []) or []:
                intent = (candidate.get("intent_candidate") or "").strip()
                scenario = (candidate.get("scenario_candidate") or "").strip()
                if intent or scenario:
                    if cluster_on_intent_only:
                        flow_keys.append(intent)
                    else:
                        flow_keys.append(f"{intent} | {scenario}".strip(" |"))
            for candidate in atoms.get("faq_candidates", []) or []:
                q_clean = (candidate.get("q_clean") or "").strip()
                if q_clean:
                    faq_questions.append(q_clean)
        if client is None:
            st.error("OPENAI_API_KEY required for embeddings.")
        else:
            with st.spinner("Generating embeddings..."):
                if flow_keys:
                    embed_texts(flow_keys, client, embed_model, embed_cache, namespace=project_id)
                if faq_questions:
                    embed_texts(faq_questions, client, embed_model, embed_cache, namespace=project_id)
            mark_stage(run_state, "embeddings")
            save_run_state(ws.root, run_state)
            st.success("Embeddings generated and cached.")

if clear_normalize:
    _remove_path(ws.normalized_path)
    st.success("Cleared normalization output.")

if clear_atoms:
    _remove_path(ws.atoms_path)
    st.success("Cleared atoms output.")

if clear_faqs:
    _remove_path(ws.faq_catalogue_path)
    st.success("Cleared FAQ output.")

if clear_intents:
    _remove_path(ws.intent_flow_catalogue_path)
    st.success("Cleared intent/flow output.")

if clear_blueprint:
    _remove_path(ws.blueprint_path)
    st.success("Cleared blueprint output.")

if run_all or run_normalize:
    raw_files = (
        list(ws.input_dir.glob("*.json"))
        + list(ws.input_dir.glob("*.jsonl"))
        + list(ws.input_dir.glob("*.txt"))
    )
    if not raw_files:
        st.warning("Upload a JSON/JSONL/TXT file first.")
    else:
        all_rows = []
        for path in raw_files:
            all_rows.extend(read_raw_records(path))
        normalized = normalize_records(all_rows, redact_pii=redact_pii)
        save_jsonl(ws.normalized_path, normalized)
        mark_stage(run_state, "normalize")
        save_run_state(ws.root, run_state)
        st.success(f"Normalized {len(normalized)} conversations.")

if run_all or run_atoms:
    if ws.normalized_path.exists():
        conversations = load_jsonl(ws.normalized_path)
        if client is None:
            st.error("OPENAI_API_KEY required for atoms extraction.")
        else:
            atoms_system = get_prompt(prompts, "atoms.system", ATOMS_SYSTEM_PROMPT)
            atoms_user = get_prompt(prompts, "atoms.user", ATOMS_USER_TEMPLATE)
            prog = st.progress(0.0)

            def _update(done: int, total: int) -> None:
                prog.progress(done / total if total else 0.0)

            atoms_rows, failures = extract_atoms(
                conversations,
                client,
                llm_model,
                llm_cache,
                system_prompt=atoms_system,
                user_template=atoms_user,
                progress_cb=_update,
            )
            save_jsonl(ws.atoms_path, atoms_rows)
            mark_stage(run_state, "atoms")
            save_run_state(ws.root, run_state)
            if failures:
                (ws.logs_dir / "atoms_failures.log").write_text("\n".join(failures))
                st.warning(f"Failed atoms extraction for {len(failures)} conversations.")
            st.success(f"Saved atoms for {len(atoms_rows)} conversations.")
    else:
        st.warning("Run normalization first.")

if run_all or run_faqs:
    if ws.atoms_path.exists():
        atoms_rows = load_jsonl(ws.atoms_path)
        if not run_state.get("embeddings"):
            st.warning("Generate embeddings before running FAQs.")
            atoms_rows = []
        elif client is None:
            st.error("OPENAI_API_KEY required for embeddings and FAQ clustering.")
        faq_system = get_prompt(prompts, "faq.system", FAQ_SYSTEM_PROMPT)
        faq_user = get_prompt(prompts, "faq.user", FAQ_USER_TEMPLATE)
        if atoms_rows:
            faq_entries = build_faq_catalogue(
                atoms_rows,
                client,
                llm_model,
                llm_cache,
                embed_model,
                embed_cache,
                embed_namespace=project_id,
                system_prompt=faq_system,
                user_template=faq_user,
                threshold=threshold,
                min_freq=min_faq_freq,
            )
            save_json(ws.faq_catalogue_path, {"faqs": faq_entries})
            mark_stage(run_state, "faqs")
            save_run_state(ws.root, run_state)
            st.success(f"Saved {len(faq_entries)} FAQs.")
    else:
        st.warning("Run atoms extraction first.")

if run_all or run_intents:
    if ws.atoms_path.exists():
        atoms_rows = load_jsonl(ws.atoms_path)
        if not run_state.get("embeddings"):
            st.warning("Generate embeddings before running intents.")
            atoms_rows = []
        elif client is None:
            st.error("OPENAI_API_KEY required for embeddings and intent clustering.")
        intent_system = get_prompt(prompts, "intent_flow.system", INTENT_SYSTEM_PROMPT)
        intent_user = get_prompt(prompts, "intent_flow.user", INTENT_USER_TEMPLATE)
        if atoms_rows:
            cluster_path = ws.root / "stage_cluster" / "flow_clusters.json"
            if not cluster_path.exists():
                st.warning("Generate clusters in the Clusters tab before building intents.")
                intents = []
            else:
                clusters_payload = load_json(cluster_path)
                flow_items = build_flow_items(atoms_rows, cluster_on_intent_only)
                flow_keys = [item["flow_key"] for item in flow_items]
                flow_keys_hash = sha256_text(json.dumps(flow_keys, ensure_ascii=True))
                payload_hash = clusters_payload.get("flow_keys_hash")
                payload_threshold = clusters_payload.get("threshold")
                payload_intent_only = clusters_payload.get("cluster_on_intent_only")
                if payload_hash != flow_keys_hash or payload_threshold != threshold or payload_intent_only != cluster_on_intent_only:
                    st.warning(
                        "Clusters are out of date. Open the Clusters tab with the current "
                        "settings to regenerate before building intents."
                    )
                    intents = []
                else:
                    clusters = clusters_payload.get("clusters", [])
                    clusters = [c for c in clusters if len(c) >= min_intent_cluster_size]
                    intents = build_intent_flow_catalogue_from_clusters(
                        flow_items,
                        clusters,
                        client,
                        llm_model,
                        llm_cache,
                        min_intent_freq,
                        intent_system,
                        intent_user,
                        max_candidates_per_cluster,
                    )
            save_json(ws.intent_flow_catalogue_path, {"intents": intents})
            mark_stage(run_state, "intents")
            save_run_state(ws.root, run_state)
            st.success(f"Saved {len(intents)} intents.")
    else:
        st.warning("Run atoms extraction first.")

if run_all or run_blueprint:
    if ws.atoms_path.exists() and ws.faq_catalogue_path.exists():
        atoms_rows = load_jsonl(ws.atoms_path)
        faq_data = load_json(ws.faq_catalogue_path)
        faq_entries = faq_data.get("faqs", [])
        if blueprint_use_llm:
            if client is None:
                st.error("OPENAI_API_KEY required for LLM blueprint generation.")
                blueprint = build_blueprint(atoms_rows, faq_entries)
            else:
                blueprint_system = get_prompt(prompts, "blueprint.system", BLUEPRINT_SYSTEM_PROMPT)
                blueprint_user = get_prompt(prompts, "blueprint.user", BLUEPRINT_USER_TEMPLATE)
                blueprint = build_blueprint_with_llm(
                    atoms_rows,
                    faq_entries,
                    client,
                    llm_model,
                    llm_cache,
                    system_prompt=blueprint_system,
                    user_template=blueprint_user,
                )
        else:
            blueprint = build_blueprint(atoms_rows, faq_entries)
        save_json(ws.blueprint_path, blueprint)
        mark_stage(run_state, "blueprint")
        save_run_state(ws.root, run_state)
        st.success("Saved blueprint.")
    else:
        st.warning("Run atoms extraction and FAQs first.")

if publish_parlant:
    if not parlant_enabled:
        st.warning("Enable Parlant in the sidebar before syncing.")
    else:
        sync_result = sync_project_to_parlant(ws.root, parlant_base_url)
        st.session_state["parlant_last_sync"] = sync_result
        if sync_result.get("ok"):
            if not sync_result.get("skipped"):
                mark_stage(run_state, "parlant_sync")
                save_run_state(ws.root, run_state)
            if sync_result.get("skipped"):
                st.info("Parlant is already in sync.")
            else:
                st.success("Parlant sync complete.")
        else:
            st.error("Parlant sync failed. Check logs for details.")


# ---------- Main UI tabs ----------

tabs = st.tabs(
    [
        "Preview",
        "Atoms",
        "FAQs",
        "Intents/Flows",
        "Clusters",
        "Blueprint",
        "Export",
        "Chatbot (Preview)",
        "Final Prompt",
        "Testing",
    ]
)

with tabs[0]:
    st.subheader("Normalized preview")
    if ws.normalized_path.exists():
        conversations = load_jsonl(ws.normalized_path)
        meta_rows = [
            {
                "conversation_id": c.get("conversation_id"),
                "num_messages": c.get("metadata", {}).get("num_messages"),
                "total_text_chars": c.get("metadata", {}).get("total_text_chars"),
                "removed_filler_count": c.get("metadata", {}).get("removed_filler_count"),
                "is_low_signal": c.get("metadata", {}).get("is_low_signal"),
            }
            for c in conversations
        ]
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)
        convo_ids = [c.get("conversation_id") for c in conversations]
        selected = st.selectbox("Conversation", convo_ids, key="preview_convo")
        convo = next((c for c in conversations if c.get("conversation_id") == selected), None)
        if convo:
            st.json(convo.get("metadata", {}))
            st.markdown("**Messages**")
            for msg in convo.get("messages", []):
                filler = " (filler)" if msg.get("is_filler") else ""
                st.write(f"[{msg.get('timestamp')}] {msg.get('role')}: {msg.get('text')}{filler}")
            st.markdown("**messages_for_llm**")
            for msg in convo.get("messages_for_llm", []):
                st.write(f"[{msg.get('timestamp')}] {msg.get('role')}: {msg.get('text')}")
    else:
        st.info("Run normalization to see preview.")

with tabs[1]:
    st.subheader("Atoms")
    if ws.atoms_path.exists():
        atoms_rows = load_jsonl(ws.atoms_path)
        st.markdown("**Filter atoms**")
        filter_key = st.text_input("Key (dot notation)", value="", placeholder="e.g., conversation_id or blueprint_signals.policy")
        filter_value = st.text_input("Value contains", value="", placeholder="e.g., billing")

        def _get_nested(d: Dict[str, Any], key: str) -> Any:
            if not key:
                return None
            cur = d
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur

        def _matches(atom: Dict[str, Any]) -> bool:
            if not filter_key or not filter_value:
                return True
            value = _get_nested(atom, filter_key)
            if value is None:
                return False
            if isinstance(value, list):
                return any(filter_value.lower() in str(v).lower() for v in value)
            return filter_value.lower() in str(value).lower()

        filtered_atoms = [a for a in atoms_rows if _matches(a)]
        summary = [
            {
                "conversation_id": a.get("conversation_id"),
                "faq_candidates": len(a.get("faq_candidates", []) or []),
                "flow_candidates": len(a.get("flow_candidates", []) or []),
                "policy": ", ".join((a.get("blueprint_signals", {}) or {}).get("policy", [])[:3]),
                "tone": ", ".join((a.get("blueprint_signals", {}) or {}).get("tone", [])[:3]),
            }
            for a in filtered_atoms
        ]
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        convo_ids = [a.get("conversation_id") for a in filtered_atoms]
        selected = st.selectbox("Atoms detail", convo_ids, key="atoms_detail")
        detail = next((a for a in filtered_atoms if a.get("conversation_id") == selected), None)
        if detail:
            st.json(detail)
            if ws.normalized_path.exists():
                conversations = load_jsonl(ws.normalized_path)
                convo = next((c for c in conversations if c.get("conversation_id") == selected), None)
                if convo:
                    st.markdown("**Transcript preview (messages_for_llm)**")
                    for msg in convo.get("messages_for_llm", [])[:12]:
                        st.write(f"[{msg.get('timestamp')}] {msg.get('role')}: {msg.get('text')}")
    else:
        st.info("Run atoms extraction to see results.")

with tabs[2]:
    st.subheader("FAQs")
    st.markdown("**Upload FAQs (CSV)**")
    st.caption("Expected columns: Question, Answer")
    faq_upload = st.file_uploader("Upload FAQ CSV", type=["csv"], key="faq_csv_upload")
    faq_upload_mode = st.radio(
        "Upload mode",
        ["Append to existing", "Replace existing"],
        horizontal=True,
        key="faq_upload_mode",
    )
    if faq_upload is not None:
        try:
            faq_df = pd.read_csv(faq_upload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
            faq_df = pd.DataFrame()
        if not faq_df.empty:
            if "Question" not in faq_df.columns or "Answer" not in faq_df.columns:
                st.error("CSV must include columns: Question, Answer")
            else:
                if st.button("Import FAQs from CSV", key="import_faq_csv"):
                    existing = []
                    if faq_upload_mode == "Append to existing" and ws.faq_catalogue_path.exists():
                        existing = load_json(ws.faq_catalogue_path).get("faqs", [])
                    start_idx = len(existing)
                    new_entries = []
                    for idx, row in faq_df.iterrows():
                        question = str(row.get("Question") or "").strip()
                        answer = str(row.get("Answer") or "").strip()
                        if not question:
                            continue
                        new_entries.append(
                            {
                                "faq_id": f"faq_{start_idx + len(new_entries) + 1}",
                                "canonical_question": question,
                                "question_variants": [],
                                "draft_answer": answer,
                                "needs_verification": False,
                                "value_score": {"0_to_5": 2.5, "reason": "uploaded_csv"},
                            }
                        )
                    save_json(ws.faq_catalogue_path, {"faqs": existing + new_entries})
                    mark_stage(run_state, "faqs")
                    save_run_state(ws.root, run_state)
                    st.success(f"Imported {len(new_entries)} FAQs from CSV.")
                st.dataframe(faq_df.head(10), use_container_width=True)
    if ws.faq_catalogue_path.exists():
        faq_data = load_json(ws.faq_catalogue_path)
        faqs = faq_data.get("faqs", [])
        scores = []
        for faq in faqs:
            value_score = faq.get("value_score") or {}
            score = value_score.get("0_to_5")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        score_min = min(scores) if scores else 0.0
        score_max = max(scores) if scores else 5.0
        score_range = st.slider(
            "Filter by value score",
            min_value=0.0,
            max_value=5.0,
            value=(score_min, score_max),
            step=0.1,
        )
        rows = []
        filtered_faqs = []
        for faq in faqs:
            value_score = faq.get("value_score") or {}
            score = value_score.get("0_to_5")
            if isinstance(score, (int, float)) and not (score_range[0] <= float(score) <= score_range[1]):
                continue
            filtered_faqs.append(faq)
            rows.append(
                {
                    "faq_id": faq.get("faq_id"),
                    "canonical_question": faq.get("canonical_question"),
                    "variants": len(faq.get("question_variants", []) or []),
                    "needs_verification": faq.get("needs_verification"),
                    "value_score": score,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        for faq in filtered_faqs:
            value_score = faq.get("value_score") or {}
            score = value_score.get("0_to_5")
            needs_review = bool(faq.get("needs_verification"))
            if score is None:
                pill_class = "pill-neutral"
                score_label = "Score: n/a"
            elif score >= 4:
                pill_class = "pill-high"
                score_label = f"Score: {score}"
            elif score >= 2.5:
                pill_class = "pill-mid"
                score_label = f"Score: {score}"
            else:
                pill_class = "pill-low"
                score_label = f"Score: {score}"
            review_badge = (
                "<span class='badge badge-review'>⚠ Review required</span>"
                if needs_review
                else "<span class='badge badge-ok'>✅ Good</span>"
            )
            variants = faq.get("question_variants", []) or []
            draft_answer = faq.get("draft_answer") or ""
            st.markdown(
                f"""
<div class="faq-card">
  <div class="faq-title">{faq.get('canonical_question', '')}</div>
  <div style="margin:8px 0;">
    <span class="pill {pill_class}">{score_label}</span>
    {review_badge}
  </div>
  <div><strong>Variants:</strong> {len(variants)}</div>
  <div style="margin-top:6px;"><strong>Draft answer:</strong> {draft_answer}</div>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info("Run FAQ stage.")

    st.divider()
    st.subheader("FAQ Clusters (q_clean)")
    if not ws.atoms_path.exists():
        st.info("Run atoms extraction to view FAQ clusters.")
    else:
        atoms_rows = load_jsonl(ws.atoms_path)
        faq_rows = collect_faq_rows(atoms_rows)
        if not faq_rows:
            st.info("No FAQ candidates found in atoms.")
        elif client is None:
            st.warning("OPENAI_API_KEY required for embeddings and FAQ clustering.")
        else:
            min_cluster_size = st.number_input(
                "Min cluster size",
                min_value=1,
                value=3,
                step=1,
                key="faq_cluster_min_size",
            )
            clusters = cluster_faq_rows(faq_rows, client, embed_model, embed_cache, threshold, namespace=project_id)
            if not clusters:
                st.info("No FAQ clusters found for the current threshold.")
            else:
                cluster_rows = []
                for idx, cluster in enumerate(clusters):
                    cluster_items = [faq_rows[i] for i in cluster]
                    samples = [row.get("q_clean") for row in cluster_items if row.get("q_clean")]
                    cluster_rows.append(
                        {
                            "cluster_id": f"faq_cluster_{idx}",
                            "size": len(cluster),
                            "sample_questions": ", ".join(list(dict.fromkeys(samples))[:3]),
                            "conversation_ids": len({c.get("conversation_id") for c in cluster_items}),
                        }
                    )
                cluster_rows = [row for row in cluster_rows if row["size"] >= min_cluster_size]
                cluster_rows = sorted(cluster_rows, key=lambda row: row["size"], reverse=True)
                st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

                cluster_ids = [row["cluster_id"] for row in cluster_rows]
                if not cluster_ids:
                    st.info("No clusters meet the minimum size filter.")
                else:
                    selected = st.selectbox("FAQ cluster", cluster_ids, key="faq_cluster_select")
                    selected_idx = int(selected.split("_")[-1])
                    selected_items = [faq_rows[i] for i in clusters[selected_idx]]
                    st.markdown("**Cluster questions**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "conversation_id": item.get("conversation_id"),
                                    "q_clean": item.get("q_clean"),
                                    "q_raw": item.get("q_raw"),
                                    "answers": len(item.get("a_candidates") or []),
                                }
                                for item in selected_items
                            ]
                        ),
                        use_container_width=True,
                    )

                    st.markdown("**Generate FAQ for this cluster**")
                    append_faq = st.checkbox(
                        "Append to FAQ catalogue",
                        value=True,
                        key="append_faq_cluster",
                    )
                    if st.button("Generate FAQs from all clusters", key="gen_faq_all_clusters"):
                        existing = []
                        if ws.faq_catalogue_path.exists():
                            existing = load_json(ws.faq_catalogue_path).get("faqs", [])
                        generated = []
                        for idx, cluster in enumerate(clusters):
                            cluster_items = [faq_rows[i] for i in cluster]
                            faq_entry = build_faq_from_cluster(
                                cluster_items,
                                client,
                                llm_model,
                                llm_cache,
                                system_prompt=get_prompt(prompts, "faq.system", FAQ_SYSTEM_PROMPT),
                                user_template=get_prompt(prompts, "faq.user", FAQ_USER_TEMPLATE),
                                faq_id=f"faq_{len(existing) + idx + 1}",
                                min_freq=min_faq_freq,
                                force_llm=True,
                            )
                            generated.append(faq_entry)
                        if append_faq:
                            save_json(ws.faq_catalogue_path, {"faqs": existing + generated})
                            mark_stage(run_state, "faqs")
                            save_run_state(ws.root, run_state)
                            st.success(f"Saved {len(generated)} FAQs to catalogue.")
                        else:
                            st.success(f"Generated {len(generated)} FAQs.")
                        st.json(generated[:5])
                    if st.button("Generate FAQ from selected cluster", key="gen_faq_cluster"):
                        existing = []
                        if ws.faq_catalogue_path.exists():
                            existing = load_json(ws.faq_catalogue_path).get("faqs", [])
                        next_id = f"faq_{len(existing) + 1}"
                        faq_entry = build_faq_from_cluster(
                            selected_items,
                            client,
                            llm_model,
                            llm_cache,
                            system_prompt=get_prompt(prompts, "faq.system", FAQ_SYSTEM_PROMPT),
                            user_template=get_prompt(prompts, "faq.user", FAQ_USER_TEMPLATE),
                            faq_id=next_id,
                            min_freq=min_faq_freq,
                            force_llm=True,
                        )
                        if append_faq:
                            merged = existing + [faq_entry]
                            save_json(ws.faq_catalogue_path, {"faqs": merged})
                            mark_stage(run_state, "faqs")
                            save_run_state(ws.root, run_state)
                            st.success("FAQ saved to catalogue.")
                        st.json(faq_entry)

with tabs[3]:
    st.subheader("Intent + Flow Catalogue")
    if ws.intent_flow_catalogue_path.exists():
        intent_data = load_json(ws.intent_flow_catalogue_path)
        intents, legacy_mode = normalize_intents(intent_data.get("intents", []))
        if legacy_mode:
            st.warning(
                "Legacy intent format detected. Mapped `confidence` → `intent_confidence`. "
                "Scenario confidence and automation suitability are unavailable for this run."
            )
        scores = []
        for intent in intents:
            intent_confidence = intent.get("intent_confidence") or {}
            score = intent_confidence.get("score_0_to_1")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        score_min = min(scores) if scores else 0.0
        score_max = max(scores) if scores else 1.0
        score_range = st.slider(
            "Filter by intent confidence score",
            min_value=0.0,
            max_value=1.0,
            value=(score_min, score_max),
            step=0.05,
        )
        rows = []
        filtered_intents = []
        for intent in intents:
            scenarios = intent.get("scenarios", []) or []
            total_freq = sum(s.get("frequency", 0) for s in scenarios)
            intent_confidence = intent.get("intent_confidence") or {}
            conf_score_val = intent_confidence.get("score_0_to_1")
            if isinstance(conf_score_val, (int, float)) and not (
                score_range[0] <= float(conf_score_val) <= score_range[1]
            ):
                continue
            filtered_intents.append(intent)
            rows.append(
                {
                    "intent_name": intent.get("intent_name"),
                    "intent_confidence": intent_confidence.get("score_0_to_1"),
                    "scenarios": len(scenarios),
                    "total_frequency": total_freq,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        intent_names = [i.get("intent_name") for i in filtered_intents]
        if not intent_names:
            st.info("No intents match the selected score filter.")
        else:
            left, right = st.columns([2, 5], gap="large")
            with left:
                st.markdown("**Intent list**")
                search_query = st.text_input("Search intents", value="", placeholder="Type to filter...")
                sort_by = st.selectbox(
                    "Sort by",
                    ["frequency", "confidence", "name"],
                    index=0,
                )
                display_intents = filtered_intents
                if search_query:
                    q = search_query.strip().lower()
                    display_intents = [
                        i for i in display_intents if q in (i.get("intent_name") or "").lower()
                    ]
                if sort_by == "frequency":
                    display_intents = sorted(
                        display_intents,
                        key=lambda i: sum(
                            s.get("frequency", 0) for s in (i.get("scenarios") or [])
                        ),
                        reverse=True,
                    )
                elif sort_by == "confidence":
                    display_intents = sorted(
                        display_intents,
                        key=lambda i: (i.get("intent_confidence") or {}).get("score_0_to_1") or 0,
                        reverse=True,
                    )
                else:
                    display_intents = sorted(
                        display_intents, key=lambda i: (i.get("intent_name") or "").lower()
                    )

                if not display_intents:
                    st.info("No intents match the search.")
                if "intent_open_idx" not in st.session_state or st.session_state["intent_open_idx"] >= len(display_intents):
                    st.session_state["intent_open_idx"] = 0
                for idx, intent in enumerate(display_intents):
                    name = intent.get("intent_name", "")
                    intent_confidence = intent.get("intent_confidence") or {}
                    score = intent_confidence.get("score_0_to_1")
                    if score is None:
                        pill_class = "pill-neutral"
                        score_label = "n/a"
                    elif score >= 0.8:
                        pill_class = "pill-high"
                        score_label = f"{score}"
                    elif score >= 0.5:
                        pill_class = "pill-mid"
                        score_label = f"{score}"
                    else:
                        pill_class = "pill-low"
                        score_label = f"{score}"
                    cols = st.columns([4, 1])
                    with cols[0]:
                        if st.button(name, key=f"intent_pick_{idx}_{name}", use_container_width=True):
                            st.session_state["intent_open_idx"] = idx
                    with cols[1]:
                        st.markdown(
                            f"<span class='pill {pill_class}'>{score_label}</span>",
                            unsafe_allow_html=True,
                        )
                selected_idx = st.session_state.get("intent_open_idx", 0)
                selected_intent = display_intents[selected_idx] if display_intents else None
            with right:
                if not selected_intent:
                    st.info("Select an intent to view details.")
                else:
                    intent = selected_intent
                    intent_confidence = intent.get("intent_confidence") or {}
                    conf_score = intent_confidence.get("score_0_to_1")
                    if conf_score is None:
                        conf_class = "pill-neutral"
                        conf_label = "Intent confidence: n/a"
                    elif conf_score >= 0.8:
                        conf_class = "pill-high"
                        conf_label = f"Intent confidence: {conf_score}"
                    elif conf_score >= 0.5:
                        conf_class = "pill-mid"
                        conf_label = f"Intent confidence: {conf_score}"
                    else:
                        conf_class = "pill-low"
                        conf_label = f"Intent confidence: {conf_score}"
                    st.markdown(
                        f"""
<div class="detail-card">
  <div class="detail-title">{intent.get('intent_name', '')}</div>
  <div class="detail-subtitle">{intent.get('definition', '')}</div>
  <div class="chip-row">
    <span class="pill {conf_class}">{conf_label}</span>
    <span class="chip">Scenarios: {len(intent.get('scenarios', []) or [])}</span>
  </div>
  <div class="section-title">Intent confidence reason</div>
  <div class="muted">{intent_confidence.get('reason', '')}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    scenarios = intent.get("scenarios", []) or []
                    if not scenarios:
                        st.info("No scenarios available for this intent.")
                    else:
                        scenario_rows = []
                        for scenario in scenarios:
                            scenario_confidence = scenario.get("scenario_confidence") or {}
                            automation = scenario.get("automation_suitability") or {}
                            scenario_rows.append(
                                {
                                    "scenario_name": scenario.get("scenario_name"),
                                    "frequency": scenario.get("frequency"),
                                    "scenario_confidence": scenario_confidence.get("score_0_to_1"),
                                    "automation_suitable": automation.get("suitable"),
                                    "automation_score": automation.get("score_0_to_1"),
                                }
                            )
                        st.markdown("**Scenarios**")
                        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True)
                        scenario_names = [s.get("scenario_name") for s in scenarios]
                        selected_scenario = st.selectbox(
                            "Scenario details",
                            scenario_names,
                            key=f"scenario_select_{intent.get('intent_name')}",
                        )
                        scenario = next((s for s in scenarios if s.get("scenario_name") == selected_scenario), None)
                        if scenario:
                            scenario_confidence = scenario.get("scenario_confidence") or {}
                            automation = scenario.get("automation_suitability") or {}
                            scenario_conf_score = scenario_confidence.get("score_0_to_1")
                            if scenario_conf_score is None:
                                scenario_conf_class = "pill-neutral"
                                scenario_conf_label = "Scenario confidence: n/a"
                            elif scenario_conf_score >= 0.8:
                                scenario_conf_class = "pill-high"
                                scenario_conf_label = f"Scenario confidence: {scenario_conf_score}"
                            elif scenario_conf_score >= 0.5:
                                scenario_conf_class = "pill-mid"
                                scenario_conf_label = f"Scenario confidence: {scenario_conf_score}"
                            else:
                                scenario_conf_class = "pill-low"
                                scenario_conf_label = f"Scenario confidence: {scenario_conf_score}"
                            auto_score = automation.get("score_0_to_1")
                            if auto_score is None:
                                auto_class = "pill-neutral"
                                auto_label = "Automation: n/a"
                            elif auto_score >= 0.7:
                                auto_class = "pill-high"
                                auto_label = f"Automation: {auto_score}"
                            elif auto_score >= 0.4:
                                auto_class = "pill-mid"
                                auto_label = f"Automation: {auto_score}"
                            else:
                                auto_class = "pill-low"
                                auto_label = f"Automation: {auto_score}"
                            suitable = automation.get("suitable")
                            suitability_badge = (
                                "<span class='badge badge-ok'>✅ Suitable</span>"
                                if suitable
                                else "<span class='badge badge-review'>⚠ Not suitable</span>"
                            )
                            st.markdown(
                                f"""
<div class="detail-card">
  <div class="detail-title">{scenario.get('scenario_name', '')}</div>
  <div class="detail-subtitle">Frequency: {scenario.get('frequency', 0)}</div>
  <div class="chip-row">
    <span class="pill {scenario_conf_class}">{scenario_conf_label}</span>
    <span class="pill {auto_class}">{auto_label}</span>
    {suitability_badge}
  </div>
  <div class="section-title">Scenario confidence reason</div>
  <div class="muted">{scenario_confidence.get('reason', '')}</div>
  <div class="section-title">Automation suitability reason</div>
  <div class="muted">{automation.get('reason', '')}</div>
</div>
""",
                                unsafe_allow_html=True,
                            )
                            flow_steps = scenario.get("flow_template", []) or []
                            if flow_steps:
                                st.markdown("<div class='flowchart-frame'>", unsafe_allow_html=True)
                                for step_idx, step in enumerate(flow_steps, start=1):
                                    step_type = step.get("type", "step")
                                    text = step.get("text", "")
                                    st.markdown(
                                        f"""
<div class="flow-step">
  <div class="type">Step {step_idx} · {step_type}</div>
  <div>{text}</div>
</div>
""",
                                        unsafe_allow_html=True,
                                    )
                                st.markdown("</div>", unsafe_allow_html=True)
                            required_inputs = scenario.get("required_inputs", []) or []
                            handover_rules = scenario.get("handover_rules", []) or []
                            if required_inputs:
                                st.markdown(
                                    f"**Required inputs:** {', '.join(required_inputs)}"
                                )
                            if handover_rules:
                                st.markdown(
                                    f"**Handover rules:** {', '.join(handover_rules)}"
                                )
                    st.divider()
                    st.subheader("Push intent to Parlant")
                    if not parlant_enabled:
                        st.info("Enable Parlant in the sidebar to push intents.")
                    elif not parlant_base_url:
                        st.warning("Set PARLANT_BASE_URL in the sidebar.")
                    else:
                        agents_state_key = f"parlant_agents_{parlant_base_url}"
                        refresh_agents = st.button(
                            "Refresh agents",
                            key=f"parlant_refresh_agents_{intent.get('intent_id')}",
                        )
                        if refresh_agents or agents_state_key not in st.session_state:
                            try:
                                agents_payload = fetch_parlant_agents(parlant_base_url)
                                st.session_state[agents_state_key] = agents_payload
                            except OpenAPIError as exc:
                                st.session_state[agents_state_key] = {"error": str(exc), "agents": []}
                        agents_payload = st.session_state.get(agents_state_key, {})
                        if agents_payload.get("error"):
                            st.error(f"Failed to load agents: {agents_payload['error']}")
                        else:
                            agents = agents_payload.get("agents", [])
                            options = [{"label": "Create new agent", "id": None}]
                            for agent in agents:
                                agent_id = extract_id(agent)
                                label = f"{agent.get('name') or 'Unnamed'} ({agent_id})"
                                options.append({"label": label, "id": agent_id})
                            option_labels = [opt["label"] for opt in options]
                            selected_label = st.selectbox(
                                "Target agent",
                                option_labels,
                                key=f"parlant_agent_select_{intent.get('intent_id')}",
                            )
                            selected = next(opt for opt in options if opt["label"] == selected_label)
                            create_agent_name = None
                            create_agent_description = None
                            if selected["id"] is None:
                                create_agent_name = st.text_input(
                                    "New agent name",
                                    value=f"{intent.get('intent_name')} Agent",
                                    key=f"parlant_agent_name_{intent.get('intent_id')}",
                                )
                                create_agent_description = st.text_input(
                                    "New agent description",
                                    value="Agent generated from intent discovery scenarios.",
                                    key=f"parlant_agent_desc_{intent.get('intent_id')}",
                                )
                            use_llm = st.checkbox(
                                "Use OpenAI to format journeys",
                                value=True,
                                key=f"parlant_use_llm_{intent.get('intent_id')}",
                            )
                            llm_system = get_prompt(
                                prompts,
                                "parlant_journey.system",
                                PARLANT_JOURNEY_SYSTEM_PROMPT,
                            )
                            llm_user = get_prompt(
                                prompts,
                                "parlant_journey.user",
                                PARLANT_JOURNEY_USER_TEMPLATE,
                            )
                            preview_key = f"parlant_preview_intent_{intent.get('intent_id')}"
                            if st.button("Preview journeys", key=preview_key):
                                if use_llm and client is None:
                                    st.error("OPENAI_API_KEY required to format journeys with OpenAI.")
                                else:
                                    with st.spinner("Generating journey previews..."):
                                        try:
                                            preview = preview_intent_journeys(
                                                intent,
                                                parlant_base_url,
                                                use_llm,
                                                client,
                                                llm_model,
                                                llm_cache,
                                                llm_system,
                                                llm_user,
                                            )
                                        except OpenAPIError as exc:
                                            preview = {"ok": False, "errors": [str(exc)], "journeys": []}
                                    st.session_state[f"parlant_preview_result_{intent.get('intent_id')}"] = preview
                            preview_result = st.session_state.get(
                                f"parlant_preview_result_{intent.get('intent_id')}"
                            )
                            if preview_result:
                                if preview_result.get("ok"):
                                    st.success(
                                        f"Prepared {len(preview_result.get('journeys', []))} journey payload(s)."
                                    )
                                if preview_result.get("errors"):
                                    st.error("Failed to build preview.")
                                    st.json(preview_result["errors"])
                                journeys = preview_result.get("journeys") or []
                                if journeys:
                                    with st.expander("Journey preview payloads", expanded=False):
                                        for entry in journeys:
                                            st.markdown(f"**{entry.get('scenario_name')}**")
                                            st.json(entry.get("payload") or {})
                            push_key = f"parlant_push_intent_{intent.get('intent_id')}"
                            if st.button("Push intent to Parlant", key=push_key):
                                if use_llm and client is None:
                                    st.error("OPENAI_API_KEY required to format journeys with OpenAI.")
                                    result = {"ok": False, "errors": ["Missing OPENAI_API_KEY."]}
                                    st.session_state[f"parlant_intent_result_{intent.get('intent_id')}"] = result
                                    st.rerun()
                                if selected["id"] is None and not (create_agent_name or "").strip():
                                    st.warning("Provide a name for the new agent.")
                                else:
                                    with st.spinner("Pushing intent to Parlant..."):
                                        result = push_intent_to_parlant(
                                            intent,
                                            parlant_base_url,
                                            selected["id"],
                                            create_agent_name,
                                            create_agent_description,
                                            use_llm,
                                            client,
                                            llm_model,
                                            llm_cache,
                                            llm_system,
                                            llm_user,
                                        )
                                    result_key = f"parlant_intent_result_{intent.get('intent_id')}"
                                    st.session_state[result_key] = result
                            result_key = f"parlant_intent_result_{intent.get('intent_id')}"
                            if result_key in st.session_state:
                                result = st.session_state[result_key]
                                if result.get("ok"):
                                    st.success(
                                        "Pushed to Parlant. "
                                        f"Agent: {result.get('agent_id')} | "
                                        f"Journeys created: {result.get('journeys_created')} | "
                                        f"Skipped: {result.get('journeys_skipped')}"
                                    )
                                else:
                                    st.error("Parlant push failed.")
                                if result.get("errors"):
                                    st.json(result["errors"])
                    st.markdown("**Intent JSON**")
                    st.json(intent)
    else:
        st.info("Run intent/flow stage.")

with tabs[4]:
    st.subheader("Flow Clusters")
    if ws.atoms_path.exists():
        atoms_rows = load_jsonl(ws.atoms_path)
        flow_items = build_flow_items(atoms_rows, cluster_on_intent_only)

        if not flow_items:
            st.info("No flow candidates found in atoms.")
        elif not run_state.get("embeddings"):
            st.warning("Generate embeddings to view clusters.")
        else:
            flow_keys = [item["flow_key"] for item in flow_items]
            if client is None:
                st.error("OPENAI_API_KEY required for embeddings and clustering.")
            else:
                embeddings, _ = embed_texts(flow_keys, client, embed_model, embed_cache, namespace=project_id)
                clusters = split_large_clusters(cluster_embeddings(embeddings, threshold), embeddings)
                if not clusters:
                    st.info("No clusters found for the current threshold.")
                else:
                    flow_keys_hash = sha256_text(json.dumps(flow_keys, ensure_ascii=True))
                    save_json(
                        ws.root / "stage_cluster" / "flow_clusters.json",
                        {
                            "threshold": threshold,
                            "cluster_on_intent_only": cluster_on_intent_only,
                            "flow_keys_hash": flow_keys_hash,
                            "clusters": clusters,
                        },
                    )
                    mark_stage(run_state, "clusters")
                    save_run_state(ws.root, run_state)
                    min_cluster_size = st.number_input(
                        "Min cluster size",
                        min_value=1,
                        value=3,
                        step=1,
                        help="Show only clusters with size >= this value.",
                    )
                    cluster_rows = []
                    for idx, cluster in enumerate(clusters):
                        cluster_items = [flow_items[i] for i in cluster]
                        intents = [c.get("intent_candidate") for c in cluster_items if c.get("intent_candidate")]
                        scenarios = [c.get("scenario_candidate") for c in cluster_items if c.get("scenario_candidate")]
                        cluster_rows.append(
                            {
                                "cluster_id": f"cluster_{idx}",
                                "size": len(cluster),
                                "sample_intents": ", ".join(list(dict.fromkeys(intents))[:3]),
                                "sample_scenarios": ", ".join(list(dict.fromkeys(scenarios))[:3]),
                                "conversation_ids": len({c.get("conversation_id") for c in cluster_items}),
                            }
                        )
                    cluster_rows = [
                        row for row in cluster_rows if row["size"] >= min_cluster_size
                    ]
                    cluster_rows = sorted(cluster_rows, key=lambda row: row["size"], reverse=True)
                    st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

                    cluster_ids = [row["cluster_id"] for row in cluster_rows]
                    if not cluster_ids:
                        st.info("No clusters meet the minimum size filter.")
                    else:
                        selected = st.selectbox("Cluster", cluster_ids, key="cluster_select")
                        selected_idx = int(selected.split("_")[-1])
                        selected_items = [flow_items[i] for i in clusters[selected_idx]]
                        convo_ids = {
                            item.get("conversation_id")
                            for item in selected_items
                            if item.get("conversation_id")
                        }
                        atoms_by_id = {row.get("conversation_id"): row for row in atoms_rows}
                        selected_atoms = [atoms_by_id[cid] for cid in convo_ids if cid in atoms_by_id]
                        st.markdown("**Flow candidates in cluster**")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "conversation_id": item.get("conversation_id"),
                                        "intent_candidate": item.get("intent_candidate"),
                                        "scenario_candidate": item.get("scenario_candidate"),
                                        "resolution": (item.get("resolution") or {}).get("summary", ""),
                                    }
                                    for item in selected_items
                                ]
                            ),
                            use_container_width=True,
                        )
                        with st.expander("Cluster data diagnostics", expanded=False):
                            total_items = len(selected_items)
                            total_atoms = len(selected_atoms)
                            with_intent = sum(1 for item in selected_items if item.get("intent_candidate"))
                            with_scenario = sum(1 for item in selected_items if item.get("scenario_candidate"))
                            faq_count = sum(len(item.get("faq_candidates") or []) for item in selected_items)
                            tone_count = sum(len((item.get("blueprint_signals") or {}).get("tone", [])) for item in selected_items)
                            st.write(
                                f"Items: {total_items} | Atoms: {total_atoms} | With intent: {with_intent} | "
                                f"With scenario: {with_scenario} | FAQ candidates: {faq_count} | "
                                f"Tone signals: {tone_count}"
                            )

                        st.divider()
                        st.subheader("Push cluster to Parlant")
                        if not parlant_enabled:
                            st.info("Enable Parlant in the sidebar to push clusters.")
                        elif not parlant_base_url:
                            st.warning("Set PARLANT_BASE_URL in the sidebar.")
                        else:
                            agents_state_key = f"parlant_agents_{parlant_base_url}"
                            refresh_agents = st.button(
                                "Refresh agents",
                                key=f"parlant_cluster_refresh_{selected}",
                            )
                            if refresh_agents or agents_state_key not in st.session_state:
                                try:
                                    agents_payload = fetch_parlant_agents(parlant_base_url)
                                    st.session_state[agents_state_key] = agents_payload
                                except OpenAPIError as exc:
                                    st.session_state[agents_state_key] = {"error": str(exc), "agents": []}
                            agents_payload = st.session_state.get(agents_state_key, {})
                            if agents_payload.get("error"):
                                st.error(f"Failed to load agents: {agents_payload['error']}")
                            else:
                                agents = agents_payload.get("agents", [])
                                options = [{"label": "Create new agent", "id": None}]
                                for agent in agents:
                                    agent_id = extract_id(agent)
                                    label = f"{agent.get('name') or 'Unnamed'} ({agent_id})"
                                    options.append({"label": label, "id": agent_id})
                                option_labels = [opt["label"] for opt in options]
                                selected_label = st.selectbox(
                                    "Target agent",
                                    option_labels,
                                    key=f"parlant_cluster_agent_select_{selected}",
                                )
                                selected_agent = next(
                                    opt for opt in options if opt["label"] == selected_label
                                )
                                create_agent_name = None
                                create_agent_description = None
                                if selected_agent["id"] is None:
                                    create_agent_name = st.text_input(
                                        "New agent name",
                                        value=f"{selected} Agent",
                                        key=f"parlant_cluster_agent_name_{selected}",
                                    )
                                    create_agent_description = st.text_input(
                                        "New agent description",
                                        value="Agent generated from intent discovery clusters.",
                                        key=f"parlant_cluster_agent_desc_{selected}",
                                    )
                                use_llm = st.checkbox(
                                    "Use OpenAI to format journeys",
                                    value=True,
                                    key=f"parlant_cluster_use_llm_{selected}",
                                )
                                llm_system = get_prompt(
                                    prompts,
                                    "parlant_cluster.system",
                                    PARLANT_CLUSTER_SYSTEM_PROMPT,
                                )
                                llm_user = get_prompt(
                                    prompts,
                                    "parlant_cluster.user",
                                    PARLANT_CLUSTER_USER_TEMPLATE,
                                )
                                preview_key = f"parlant_cluster_preview_{selected}"
                                if st.button("Preview Parlant candidates", key=preview_key):
                                    if use_llm and client is None:
                                        st.error("OPENAI_API_KEY required to format journeys with OpenAI.")
                                    else:
                                        with st.spinner("Generating Parlant candidates..."):
                                            try:
                                                preview = preview_cluster_candidates(
                                                    selected_atoms,
                                                    selected_items,
                                                    parlant_base_url,
                                                    use_llm,
                                                    client,
                                                    llm_model,
                                                    llm_cache,
                                                    llm_system,
                                                    llm_user,
                                                )
                                            except OpenAPIError as exc:
                                                preview = {"ok": False, "errors": [str(exc)]}
                                        st.session_state[f"parlant_cluster_preview_result_{selected}"] = preview
                                        parlant_cluster_cache[selected] = preview
                                        save_parlant_cluster_cache(ws.root, parlant_cluster_cache)
                                preview_result = st.session_state.get(
                                    f"parlant_cluster_preview_result_{selected}"
                                )
                                if not preview_result:
                                    preview_result = parlant_cluster_cache.get(selected)
                                if preview_result:
                                    if preview_result.get("ok"):
                                        st.success(
                                            f"Journeys: {len(preview_result.get('journeys') or [])} | "
                                            f"Guidelines: {len(preview_result.get('guidelines') or [])} | "
                                            f"Nodes: {len(preview_result.get('journey_nodes') or [])} | "
                                            f"Edges: {len(preview_result.get('journey_edges') or [])}"
                                        )
                                    if preview_result.get("errors"):
                                        st.error("Failed to build preview.")
                                        st.json(preview_result["errors"])
                                        st.caption("Edit the prompt in the sidebar: Parlant -> Edit Parlant cluster prompt.")
                                    if preview_result.get("journeys"):
                                        with st.expander("Journey payloads", expanded=False):
                                            for entry in preview_result.get("journeys", []):
                                                st.markdown(f"**{entry.get('scenario_name')}**")
                                                st.json(entry.get("payload") or {})
                                    if preview_result.get("guidelines"):
                                        with st.expander("Guideline payloads", expanded=False):
                                            for payload in preview_result.get("guidelines", []):
                                                st.json(payload)
                                    if preview_result.get("journey_nodes"):
                                        with st.expander("Journey node payloads", expanded=False):
                                            for entry in preview_result.get("journey_nodes", []):
                                                st.markdown(
                                                    f"**{entry.get('journey_title') or 'journey'} · {entry.get('label') or 'node'}**"
                                                )
                                                st.json(entry.get("payload") or {})
                                    if preview_result.get("journey_edges"):
                                        with st.expander("Journey edge payloads", expanded=False):
                                            for entry in preview_result.get("journey_edges", []):
                                                st.markdown(
                                                    f"**{entry.get('journey_title') or 'journey'} · {entry.get('source_label')} → {entry.get('target_label')}**"
                                                )
                                                st.json(entry.get("payload") or {})
                                    if preview_result.get("journey_nodes") and preview_result.get("journey_edges"):
                                        with st.expander("Mermaid preview", expanded=False):
                                            diagrams = build_mermaid_previews(
                                                preview_result.get("journey_nodes") or [],
                                                preview_result.get("journey_edges") or [],
                                            )
                                            show_source = st.checkbox(
                                                "Show Mermaid source",
                                                value=False,
                                                key=f"mermaid_source_{selected}",
                                            )
                                            mermaid_height = st.number_input(
                                                "Mermaid height",
                                                min_value=400,
                                                max_value=2000,
                                                value=900,
                                                step=100,
                                                key=f"mermaid_height_{selected}",
                                            )
                                            for title, diagram in diagrams.items():
                                                st.markdown(f"**{title}**")
                                                render_mermaid(diagram, height=int(mermaid_height))
                                                if show_source:
                                                    st.code(diagram, language="text")
                                push_key = f"parlant_cluster_push_{selected}"
                                if st.button("Push cluster to Parlant", key=push_key):
                                    if use_llm and client is None:
                                        st.error("OPENAI_API_KEY required to format journeys with OpenAI.")
                                    else:
                                        with st.spinner("Pushing cluster to Parlant..."):
                                            result = push_cluster_candidates(
                                                selected_atoms,
                                                selected_items,
                                                parlant_base_url,
                                                selected_agent["id"],
                                                create_agent_name,
                                                create_agent_description,
                                                use_llm,
                                                client,
                                                llm_model,
                                                llm_cache,
                                                llm_system,
                                                llm_user,
                                            )
                                        st.session_state[f"parlant_cluster_result_{selected}"] = result
                                result = st.session_state.get(f"parlant_cluster_result_{selected}")
                                if result:
                                    if result.get("ok"):
                                        st.success(
                                            "Pushed to Parlant. "
                                            f"Agent: {result.get('agent_id')} | "
                                            f"Journeys created: {result.get('journeys_created')} | "
                                            f"Guidelines created: {result.get('guidelines_created')}"
                                        )
                                    else:
                                        st.error("Parlant push failed.")
                                    if result.get("errors"):
                                        st.json(result["errors"])
                        cluster_payload = {
                            "cluster_id": selected,
                            "threshold": threshold,
                            "cluster_on_intent_only": cluster_on_intent_only,
                            "items": selected_items,
                        }
                        st.download_button(
                            "Download cluster JSON",
                            data=json.dumps(cluster_payload, ensure_ascii=True, indent=2),
                            file_name=f"{selected}.json",
                            mime="application/json",
                        )

                        st.markdown("**Generate intent for this cluster**")
                        append_intent = st.checkbox(
                            "Append to intent catalogue",
                            value=True,
                            key="append_intent_cluster",
                        )
                        if st.button("Generate intent from selected cluster", key="gen_intent_cluster"):
                            if client is None:
                                st.error("OPENAI_API_KEY required to generate intents.")
                            else:
                                cluster_intents = build_intent_flow_catalogue_from_clusters(
                                    flow_items,
                                    [clusters[selected_idx]],
                                    client,
                                    llm_model,
                                    llm_cache,
                                    min_intent_freq,
                                    get_prompt(prompts, "intent_flow.system", INTENT_SYSTEM_PROMPT),
                                    get_prompt(prompts, "intent_flow.user", INTENT_USER_TEMPLATE),
                                    max_candidates_per_cluster,
                                )
                                if append_intent:
                                    existing = []
                                    if ws.intent_flow_catalogue_path.exists():
                                        existing = load_json(ws.intent_flow_catalogue_path).get("intents", [])
                                    merged = existing + cluster_intents
                                    save_json(ws.intent_flow_catalogue_path, {"intents": merged})
                                    mark_stage(run_state, "intents")
                                    save_run_state(ws.root, run_state)
                                    st.session_state["intent_update_notice"] = (
                                        f"Appended {len(cluster_intents)} intent(s)."
                                    )
                                else:
                                    save_json(ws.intent_flow_catalogue_path, {"intents": cluster_intents})
                                    mark_stage(run_state, "intents")
                                    save_run_state(ws.root, run_state)
                                    st.session_state["intent_update_notice"] = (
                                        f"Saved {len(cluster_intents)} intent(s)."
                                    )
                                st.rerun()

                        if ws.normalized_path.exists():
                            conversations = load_jsonl(ws.normalized_path)
                            convo_map = {c.get("conversation_id"): c for c in conversations}
                            convo_ids = [
                                item.get("conversation_id")
                                for item in selected_items
                                if item.get("conversation_id")
                            ]
                            convo_ids = list(dict.fromkeys(convo_ids))
                            if convo_ids:
                                selected_convo = st.selectbox(
                                    "Conversation preview",
                                    convo_ids,
                                    key="cluster_convo_preview",
                                )
                                convo = convo_map.get(selected_convo)
                                if convo:
                                    st.markdown("**Transcript preview (messages_for_llm)**")
                                    for msg in convo.get("messages_for_llm", [])[:12]:
                                        st.write(
                                            f"[{msg.get('timestamp')}] {msg.get('role')}: {msg.get('text')}"
                                        )
                        else:
                            st.info("Run normalization to preview transcripts.")
    else:
        st.info("Run atoms extraction to explore clusters.")

with tabs[5]:
    st.subheader("Blueprint")
    if ws.blueprint_path.exists():
        blueprint = load_json(ws.blueprint_path)
        st.json(blueprint)
    else:
        st.info("Run blueprint stage.")

with tabs[6]:
    st.subheader("Export artifacts")
    if ws.root.exists():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in ws.root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ws.root))
        st.download_button(
            "Download workspace zip",
            data=buffer.getvalue(),
            file_name=f"{project_id}_artifacts.zip",
            mime="application/zip",
        )
    else:
        st.info("No artifacts to export yet.")

with tabs[7]:
    st.subheader("Chatbot (Preview)")
    init_state(st.session_state)
    assets_key = f"chatbot_assets_{project_id}"
    info_key = f"chatbot_assets_info_{project_id}"
    st.session_state.setdefault(assets_key, {})
    st.session_state.setdefault(info_key, {})

    st.markdown("### Asset Loader")
    faq_files = list_artifact_files(ws.root / "stage_faq")
    intent_files = list_artifact_files(ws.root / "stage_catalogue")
    blueprint_files = list_artifact_files(ws.root / "stage_blueprint")

    st.markdown("**Upload FAQs for chat (CSV)**")
    st.caption("Columns: Question, Answer, Tags (optional)")
    faq_chat_upload = st.file_uploader("Upload FAQ CSV for chatbot", type=["csv"], key="chatbot_faq_csv_upload")
    if faq_chat_upload is not None:
        try:
            faq_chat_df = pd.read_csv(faq_chat_upload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
            faq_chat_df = pd.DataFrame()
        if not faq_chat_df.empty:
            if "Question" not in faq_chat_df.columns or "Answer" not in faq_chat_df.columns:
                st.error("CSV must include columns: Question, Answer. Tags is optional.")
            else:
                if st.button("Load CSV FAQs into chatbot", key="chatbot_load_csv_faqs"):
                    uploaded_faqs = []
                    for idx, row in faq_chat_df.iterrows():
                        question = str(row.get("Question") or "").strip()
                        answer = str(row.get("Answer") or "").strip()
                        tags_raw = row.get("Tags") if "Tags" in faq_chat_df.columns else ""
                        tags = []
                        if isinstance(tags_raw, str):
                            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                        if not question:
                            continue
                        uploaded_faqs.append(
                            {
                                "faq_id": f"uploaded_faq_{idx + 1}",
                                "canonical_question": question,
                                "question_variants": [],
                                "answer": answer,
                                "needs_verification": False,
                                "confidence": "high",
                                "tags": tags,
                            }
                        )
                    st.session_state["chatbot_uploaded_faqs"] = uploaded_faqs
                    if client is None:
                        st.warning("OPENAI_API_KEY required to build embeddings for the chatbot.")
                    else:
                        assets = st.session_state.get(assets_key, {})
                        info = st.session_state.get(info_key, {})
                        assets["faq_index"] = build_faq_index(uploaded_faqs, client, embed_model)
                        info["faq_count"] = len(assets["faq_index"]["items"])
                        st.session_state[assets_key] = assets
                        st.session_state[info_key] = info
                        st.success(f"Loaded {len(uploaded_faqs)} FAQs into chatbot (session only).")
                st.dataframe(faq_chat_df.head(10), use_container_width=True)

    faq_options = [str(p) for p in faq_files] or ["(missing)"]
    intent_options = [str(p) for p in intent_files] or ["(missing)"]
    blueprint_options = [str(p) for p in blueprint_files] or ["(missing)"]

    selected_faq = st.selectbox("FAQ catalogue file", faq_options, key="chatbot_faq_file")
    selected_intent = st.selectbox("Intent/Flow catalogue file", intent_options, key="chatbot_intent_file")
    selected_blueprint = st.selectbox("Blueprint file", blueprint_options, key="chatbot_blueprint_file")

    thresholds_cols = st.columns(2)
    with thresholds_cols[0]:
        faq_threshold = st.slider("FAQ threshold", 0.5, 0.95, 0.82, 0.01, key="chatbot_faq_threshold")
    with thresholds_cols[1]:
        scenario_threshold = st.slider(
            "Scenario threshold", 0.5, 0.95, 0.78, 0.01, key="chatbot_scenario_threshold"
        )

    runtime_mode = st.radio(
        "Runtime mode",
        ["Native", "Parlant"],
        horizontal=True,
        key="chatbot_runtime_mode",
    )

    if "chatbot_prompt_template" not in st.session_state:
        st.session_state["chatbot_prompt_template"] = (
            "You are the chatbot preview agent.\n\n"
            "User message:\n{user_message}\n\n"
            "Mode selected: {mode_selected}\n\n"
            "Retrieved FAQs:\n{faq_matches}\n\n"
            "Retrieved Scenarios:\n{scenario_matches}\n\n"
            "Active flow state:\n{flow_state}\n\n"
            "Blueprint:\n{blueprint}\n\n"
            "Base response:\n{base_response}\n\n"
            "Respond to the user using the provided context."
        )

    if runtime_mode == "Native":
        use_llm_response = st.checkbox(
            "Use LLM to craft response (uses prompt below)", value=False, key="chatbot_use_llm"
        )
        st.text_area(
            "Chatbot prompt",
            key="chatbot_prompt_template",
            height=240,
        )
    else:
        use_llm_response = False
        st.caption("Parlant mode uses the Parlant runtime and ignores the local prompt.")

    reload_assets = st.button("Reload assets", key="chatbot_reload_assets")
    if reload_assets:
        assets = {}
        info = {"faq_count": 0, "scenario_count": 0, "blueprint_loaded": False}
        if client is None:
            st.warning("OPENAI_API_KEY required to build in-memory indexes.")
        else:
            uploaded_faqs = st.session_state.get("chatbot_uploaded_faqs") or []
            if uploaded_faqs:
                assets["faq_index"] = build_faq_index(uploaded_faqs, client, embed_model)
                info["faq_count"] = len(assets["faq_index"]["items"])
            elif selected_faq != "(missing)":
                faq_path = Path(selected_faq)
                if faq_path.exists():
                    faq_payload = load_json(faq_path)
                    assets["faq_index"] = build_faq_index(faq_payload, client, embed_model)
                    info["faq_count"] = len(assets["faq_index"]["items"])
                else:
                    st.warning("FAQ file missing; FAQ mode disabled.")
            else:
                st.warning("FAQ file missing; FAQ mode disabled.")

            if selected_intent != "(missing)":
                intent_path = Path(selected_intent)
                if intent_path.exists():
                    intent_payload = load_json(intent_path)
                    assets["scenario_index"] = build_scenario_index(intent_payload, client, embed_model)
                    info["scenario_count"] = len(assets["scenario_index"]["items"])
                else:
                    st.warning("Intent/flow file missing; Flow mode disabled.")
            else:
                st.warning("Intent/flow file missing; Flow mode disabled.")

        if selected_blueprint != "(missing)":
            blueprint_path = Path(selected_blueprint)
            if blueprint_path.exists():
                assets["blueprint"] = load_json(blueprint_path)
                info["blueprint_loaded"] = True
            else:
                st.warning("Blueprint file missing; using neutral tone.")
        else:
            st.warning("Blueprint file missing; using neutral tone.")

        st.session_state[assets_key] = assets
        st.session_state[info_key] = info

    assets = st.session_state.get(assets_key, {})
    info = st.session_state.get(info_key, {})

    status_cols = st.columns(3)
    status_cols[0].metric("FAQs loaded", info.get("faq_count", 0))
    status_cols[1].metric("Scenarios loaded", info.get("scenario_count", 0))
    status_cols[2].metric("Blueprint loaded", "yes" if info.get("blueprint_loaded") else "no")

    faq_index = assets.get("faq_index")
    scenario_index = assets.get("scenario_index")
    blueprint = assets.get("blueprint")
    native_ready = client is not None and (faq_index or scenario_index)
    parlant_ready = parlant_enabled and bool(parlant_base_url)
    allow_chat = native_ready if runtime_mode == "Native" else parlant_ready

    if not faq_index and selected_faq == "(missing)":
        st.warning("FAQ catalogue not loaded; FAQ mode disabled.")
    if not scenario_index and selected_intent == "(missing)":
        st.warning("Intent/flow catalogue not loaded; Flow mode disabled.")
    if not blueprint:
        st.warning("Blueprint not loaded; tone defaults to neutral.")
    if runtime_mode == "Native" and client is None:
        st.warning("OPENAI_API_KEY required to use chatbot preview.")
    if runtime_mode == "Parlant" and not parlant_enabled:
        st.warning("Parlant is disabled. Enable it in the sidebar to use Parlant mode.")
    if runtime_mode == "Parlant" and not parlant_base_url:
        st.warning("PARLANT_BASE_URL is required for Parlant mode.")
    if runtime_mode == "Parlant" and parlant_enabled and parlant_base_url:
        last_check = st.session_state.get("parlant_last_check_ts", 0.0)
        if time.time() - last_check > 30:
            result = test_parlant_connection(parlant_base_url)
            st.session_state["parlant_connection_ok"] = result.get("ok", False)
            st.session_state["parlant_connection_msg"] = result.get("error", "")
            st.session_state["parlant_last_check_ts"] = time.time()
        if st.session_state.get("parlant_connection_ok") is False:
            st.warning("Parlant server not running. Start with: python parlant_server/main.py")

    st.divider()
    if runtime_mode == "Parlant":
        last_sync = st.session_state.get("parlant_last_sync") or {}
        session_id = st.session_state.get(f"parlant_session_{project_id}")
        status_cols = st.columns(2)
        status_cols[0].metric("Parlant session", session_id or "not started")
        status_cols[1].metric(
            "Last sync",
            "ok" if last_sync.get("ok") else ("skipped" if last_sync.get("skipped") else "unknown"),
        )
    st.markdown("### Chat Playground")
    chat_col, debug_col = st.columns([3, 2], gap="large")
    with chat_col:
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        input_cols = st.columns([4, 1])
        with input_cols[0]:
            if st.session_state.get("chatbot_clear_input"):
                st.session_state["chatbot_user_input"] = ""
                st.session_state["chatbot_clear_input"] = False
            user_text = st.text_input(
                "Your message",
                key="chatbot_user_input",
                disabled=not allow_chat,
            )
        with input_cols[1]:
            send = st.button("Send", disabled=not allow_chat)

        reset = st.button("Reset chat", disabled=not allow_chat)
        if reset:
            st.session_state["chat_messages"] = []
            st.session_state["debug_history"] = []
            st.session_state["active_intent"] = None
            st.session_state["active_scenario"] = None
            st.session_state["step_index"] = 0
            st.session_state["collected_fields"] = {}
            st.session_state["pending_ask"] = None
            st.session_state["pending_ask_type"] = None
            st.session_state["last_debug"] = None
            st.session_state["chatbot_clear_input"] = True

        if send and user_text:
            st.session_state["chat_messages"].append({"role": "user", "content": user_text})
            if runtime_mode == "Parlant":
                try:
                    parlant_client = get_parlant_client(parlant_base_url)
                    if not parlant_client:
                        raise ParlantError("Parlant base URL is missing.")
                    runtime = ParlantRuntime(st.session_state, parlant_client)
                    responses, debug = runtime.send_message(project_id, user_text)
                except ParlantError as exc:
                    st.error(f"Parlant error: {exc}. Falling back to Native mode.")
                    if native_ready:
                        responses, debug = handle_user_message(
                            user_text,
                            {"faq_index": faq_index, "scenario_index": scenario_index},
                            client,
                            embed_model,
                            faq_threshold,
                            scenario_threshold,
                            use_llm_response,
                            llm_model,
                            st.session_state.get("chatbot_prompt_template", ""),
                            st.session_state,
                            blueprint,
                        )
                    else:
                        responses, debug = ["Parlant is unavailable and Native mode is not ready."], {
                            "mode_selected": "fallback"
                        }
            else:
                responses, debug = handle_user_message(
                    user_text,
                    {"faq_index": faq_index, "scenario_index": scenario_index},
                    client,
                    embed_model,
                    faq_threshold,
                    scenario_threshold,
                    use_llm_response,
                    llm_model,
                    st.session_state.get("chatbot_prompt_template", ""),
                    st.session_state,
                    blueprint,
                )
            st.session_state["last_debug"] = debug
            for resp in responses:
                st.session_state["chat_messages"].append({"role": "assistant", "content": resp})
                st.session_state["debug_history"].append(debug)
            st.session_state["chatbot_clear_input"] = True
            st.rerun()

    with debug_col:
        st.markdown("### Debug Panel")
        if not st.session_state.get("debug_history"):
            st.info("Debug decisions will appear here after assistant replies.")
        else:
            for idx, debug in enumerate(st.session_state["debug_history"], start=1):
                with st.expander(f"Assistant turn {idx}"):
                    st.json(debug)

with tabs[8]:
    st.subheader("Final Prompt")
    edit_final_prompt = st.button("Edit generation prompt", key="edit_final_prompt_template")
    if edit_final_prompt:
        edit_single_prompt_dialog(
            ws.root,
            prompts,
            "final_prompt.user",
            "Edit Final Prompt Template",
            FINAL_PROMPT_USER_TEMPLATE,
        )

    if not ws.intent_flow_catalogue_path.exists():
        st.info("Run intent/flow stage to generate intents.")
    else:
        intent_data = load_json(ws.intent_flow_catalogue_path)
        intents, _legacy_mode = normalize_intents(intent_data.get("intents", []))
        if not intents:
            st.info("No intents found in the catalogue.")
        else:
            intent_options = []
            for idx, intent in enumerate(intents):
                intent_id = intent.get("intent_id") or f"intent_{idx}"
                intent_name = intent.get("intent_name") or "Unnamed intent"
                label = f"{intent_name} ({intent_id})"
                intent_options.append({"label": label, "id": intent_id, "intent": intent})

            selected_label = st.selectbox(
                "Select intent",
                [opt["label"] for opt in intent_options],
                key="final_prompt_intent_select",
            )
            selected_opt = next(opt for opt in intent_options if opt["label"] == selected_label)
            selected_intent = selected_opt["intent"]
            intent_id = selected_opt["id"]

            with st.expander("Intent JSON", expanded=False):
                st.json(selected_intent)

            prompts_payload = {"prompts": {}}
            if ws.system_prompts_path.exists():
                try:
                    prompts_payload = load_json(ws.system_prompts_path)
                except json.JSONDecodeError:
                    prompts_payload = {"prompts": {}}
            saved_entry = (prompts_payload.get("prompts") or {}).get(intent_id, {})
            saved_prompt = saved_entry.get("prompt", "")
            saved_at = saved_entry.get("updated_at")
            if saved_at:
                st.caption(f"Last saved: {saved_at}")

            prompt_key = f"final_prompt_text_{intent_id}"
            pending_key = f"{prompt_key}__pending"
            if pending_key in st.session_state:
                st.session_state[prompt_key] = st.session_state.pop(pending_key)
            if prompt_key not in st.session_state:
                st.session_state[prompt_key] = saved_prompt

            prompt_text = st.text_area(
                "System prompt",
                key=prompt_key,
                height=320,
                placeholder="Generate or paste a system prompt for this intent.",
            )

            gen_col, save_col = st.columns(2)
            with gen_col:
                if st.button("Generate system prompt", key=f"final_prompt_generate_{intent_id}"):
                    if client is None:
                        st.error("OPENAI_API_KEY required to generate a system prompt.")
                    else:
                        template = get_prompt(
                            prompts,
                            "final_prompt.user",
                            FINAL_PROMPT_USER_TEMPLATE,
                        )
                        intent_json = json.dumps(selected_intent, ensure_ascii=True, indent=2)
                        try:
                            user_prompt = template.format(intent_json=intent_json)
                        except KeyError as exc:
                            st.error(f"Prompt template is missing placeholder: {exc}")
                            user_prompt = ""
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Invalid prompt template: {exc}")
                            user_prompt = ""
                        if user_prompt:
                            cache_key = llm_cache.key_for(llm_model, FINAL_PROMPT_SYSTEM, user_prompt)
                            cached = llm_cache.get(cache_key)
                            if cached is None:
                                with st.spinner("Generating system prompt..."):
                                    try:
                                        raw = with_retries(
                                            lambda: call_chat(
                                                client,
                                                llm_model,
                                                FINAL_PROMPT_SYSTEM,
                                                user_prompt,
                                                temperature=0.2,
                                            )
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        st.error(f"Generation failed: {exc}")
                                        raw = ""
                                if raw:
                                    llm_cache.set(cache_key, raw)
                                    cached = raw
                            if cached:
                                st.session_state[pending_key] = str(cached).strip()
                                st.rerun()

            with save_col:
                if st.button("Save system prompt", key=f"final_prompt_save_{intent_id}"):
                    prompt_text = (st.session_state.get(prompt_key) or "").strip()
                    if not prompt_text:
                        st.warning("System prompt is empty.")
                    else:
                        ensure_dir(ws.system_prompts_path.parent)
                        payload = {"prompts": {}}
                        if ws.system_prompts_path.exists():
                            try:
                                payload = load_json(ws.system_prompts_path)
                            except json.JSONDecodeError:
                                payload = {"prompts": {}}
                        payload_prompts = payload.get("prompts") or {}
                        payload_prompts[intent_id] = {
                            "intent_id": intent_id,
                            "intent_name": selected_intent.get("intent_name"),
                            "prompt": prompt_text,
                            "updated_at": datetime.utcnow().isoformat() + "Z",
                        }
                        payload["prompts"] = payload_prompts
                        save_json(ws.system_prompts_path, payload)
                        st.success("Saved system prompt.")

with tabs[9]:
    st.subheader("Testing")
    st.caption("Runs 5 simulated conversations using the saved system prompt and scores them with a judge model.")

    sim_model = st.text_input("Simulation model", key="simulation_model")
    judge_model = st.text_input("Judge model", key="judge_model")
    if project_id:
        ui_state.setdefault("projects", {})
        ui_state["projects"].setdefault(project_id, {})
        ui_state["projects"][project_id]["simulation_model"] = sim_model
        ui_state["projects"][project_id]["judge_model"] = judge_model
        save_ui_state(ui_state)

    cluster_path = ws.root / "stage_cluster" / "flow_clusters.json"
    if not ws.intent_flow_catalogue_path.exists():
        st.info("Run intent/flow stage to generate intents.")
    elif not ws.system_prompts_path.exists():
        st.info("Generate and save a system prompt in the Final Prompt tab first.")
    elif not ws.atoms_path.exists():
        st.info("Run atoms extraction first.")
    elif not ws.normalized_path.exists():
        st.info("Run normalization to load transcripts.")
    elif not cluster_path.exists():
        st.info("Generate clusters in the Clusters tab first.")
    else:
        intent_data = load_json(ws.intent_flow_catalogue_path)
        intents, _legacy_mode = normalize_intents(intent_data.get("intents", []))
        if not intents:
            st.info("No intents found in the catalogue.")
        else:
            intent_options = []
            for idx, intent in enumerate(intents):
                intent_id = intent.get("intent_id") or f"intent_{idx}"
                intent_name = intent.get("intent_name") or "Unnamed intent"
                label = f"{intent_name} ({intent_id})"
                intent_options.append({"label": label, "id": intent_id, "intent": intent})

            selected_label = st.selectbox(
                "Select intent",
                [opt["label"] for opt in intent_options],
                key="testing_intent_select",
            )
            selected_opt = next(opt for opt in intent_options if opt["label"] == selected_label)
            selected_intent = selected_opt["intent"]
            intent_id = selected_opt["id"]

            prompts_payload = {"prompts": {}}
            if ws.system_prompts_path.exists():
                try:
                    prompts_payload = load_json(ws.system_prompts_path)
                except json.JSONDecodeError:
                    prompts_payload = {"prompts": {}}
            saved_entry = (prompts_payload.get("prompts") or {}).get(intent_id, {})
            system_prompt_text = (saved_entry.get("prompt") or "").strip()
            can_run_tests = True
            if not system_prompt_text:
                st.warning("No saved system prompt for this intent. Use Final Prompt tab to save one.")
                can_run_tests = False

            with st.expander("System prompt", expanded=False):
                st.code(system_prompt_text, language="text")

            cluster_payload = load_json(cluster_path)
            clusters = cluster_payload.get("clusters") or []
            cluster_on_intent_only = cluster_payload.get("cluster_on_intent_only", True)

            atoms_rows = load_jsonl(ws.atoms_path)
            flow_items = build_flow_items(atoms_rows, cluster_on_intent_only)
            flow_keys = [item["flow_key"] for item in flow_items]
            current_hash = sha256_text(json.dumps(flow_keys, ensure_ascii=True))
            if current_hash != cluster_payload.get("flow_keys_hash"):
                st.warning("Clusters appear out of date with current atoms/settings. Results may be mismatched.")

            cluster_idx = _pick_cluster_for_intent(flow_items, clusters, selected_intent)
            if cluster_idx is None or cluster_idx >= len(clusters):
                st.warning("Unable to map intent to a cluster. Rebuild clusters and intents.")
                can_run_tests = False

            cluster_convo_ids: List[str] = []
            convo_map = {}
            sample_ids: List[str] = []
            if can_run_tests:
                cluster = clusters[cluster_idx]
                cluster_convo_ids = [
                    flow_items[i].get("conversation_id")
                    for i in cluster
                    if i < len(flow_items) and flow_items[i].get("conversation_id")
                ]
                cluster_convo_ids = list(dict.fromkeys(cluster_convo_ids))
                conversations = load_jsonl(ws.normalized_path)
                convo_map = {c.get("conversation_id"): c for c in conversations}
                sample_ids = [cid for cid in cluster_convo_ids if cid in convo_map][:5]
                st.write(
                    f"Cluster: cluster_{cluster_idx} | "
                    f"Conversations in cluster: {len(cluster_convo_ids)} | "
                    f"Sampled: {len(sample_ids)}"
                )

                if not sample_ids:
                    st.warning("No conversations available for testing in this cluster.")
                    can_run_tests = False

            run_tests = st.button("Run 5 simulations", key=f"testing_run_{intent_id}", disabled=not can_run_tests)
            if run_tests:
                if client is None:
                    st.error("OPENAI_API_KEY required to run simulations.")
                else:
                    results = []
                    progress = st.progress(0)
                    total = len(sample_ids)
                    for idx, convo_id in enumerate(sample_ids, start=1):
                        convo = convo_map.get(convo_id, {})
                        messages = convo.get("messages_for_llm") or []
                        user_msgs = [
                            m.get("text")
                            for m in messages
                            if m.get("role") == "customer" and (m.get("text") or "").strip()
                        ]
                        user_msgs = user_msgs[:8]
                        if not user_msgs:
                            progress.progress(idx / total)
                            continue

                        sim_messages: List[Dict[str, str]] = []
                        for user_msg in user_msgs:
                            sim_messages.append({"role": "user", "content": user_msg})
                            try:
                                response = with_retries(
                                    lambda: _call_chat_messages(
                                        client,
                                        sim_model,
                                        system_prompt_text,
                                        sim_messages,
                                        temperature=0.2,
                                    )
                                )
                            except Exception as exc:  # noqa: BLE001
                                response = f"[simulation_error] {exc}"
                            sim_messages.append({"role": "assistant", "content": response})

                        original_transcript = _format_transcript(messages)
                        simulated_transcript = _format_transcript(sim_messages)
                        intent_json = json.dumps(selected_intent, ensure_ascii=True, indent=2)
                        judge_prompt = JUDGE_USER_TEMPLATE.format(
                            intent_json=intent_json,
                            system_prompt=system_prompt_text,
                            original_transcript=original_transcript,
                            simulated_transcript=simulated_transcript,
                        )
                        judge_key = llm_cache.key_for(judge_model, JUDGE_SYSTEM_PROMPT, judge_prompt)
                        try:
                            judge_result = chat_json_with_cache(
                                client,
                                judge_model,
                                JUDGE_SYSTEM_PROMPT,
                                judge_prompt,
                                llm_cache,
                                judge_key,
                            )
                        except Exception as exc:  # noqa: BLE001
                            judge_result = {"scores": {}, "rationales": {}, "overall_notes": str(exc)}

                        score_fields = [
                            "task_completion",
                            "required_fields",
                            "flow_adherence",
                            "response_quality",
                            "escalation_handling",
                        ]
                        scores = judge_result.get("scores") or {}
                        values = []
                        for field in score_fields:
                            value = scores.get(field)
                            if isinstance(value, (int, float)):
                                values.append(float(value))
                        avg_score = sum(values) / len(score_fields) if len(values) == len(score_fields) else None

                        results.append(
                            {
                                "conversation_id": convo_id,
                                "user_messages": user_msgs,
                                "simulated_messages": sim_messages,
                                "scores": scores,
                                "rationales": judge_result.get("rationales") or {},
                                "overall_notes": judge_result.get("overall_notes") or "",
                                "average_score": avg_score,
                            }
                        )
                        progress.progress(idx / total)
                    progress.empty()

                    valid_avgs = [r["average_score"] for r in results if isinstance(r["average_score"], float)]
                    overall_avg = sum(valid_avgs) / len(valid_avgs) if valid_avgs else None

                    run_payload = {
                        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "intent_id": intent_id,
                        "intent_name": selected_intent.get("intent_name"),
                        "cluster_id": f"cluster_{cluster_idx}",
                        "simulation_model": sim_model,
                        "judge_model": judge_model,
                        "system_prompt": system_prompt_text,
                        "average_score": overall_avg,
                        "conversations": results,
                    }
                    existing_runs = {"runs": []}
                    if ws.system_prompt_tests_path.exists():
                        try:
                            existing_runs = load_json(ws.system_prompt_tests_path)
                        except json.JSONDecodeError:
                            existing_runs = {"runs": []}
                    existing_runs["runs"] = (existing_runs.get("runs") or []) + [run_payload]
                    save_json(ws.system_prompt_tests_path, existing_runs)
                    st.session_state["system_prompt_test_last_run"] = run_payload
                    st.success("Test run complete.")

            last_run = st.session_state.get("system_prompt_test_last_run")
            if not last_run and ws.system_prompt_tests_path.exists():
                try:
                    saved = load_json(ws.system_prompt_tests_path)
                    if (saved.get("runs") or []):
                        last_run = saved.get("runs")[-1]
                except json.JSONDecodeError:
                    last_run = None

            if last_run and last_run.get("intent_id") == intent_id:
                st.divider()
                st.subheader("Last test run")
                avg_display = last_run.get("average_score")
                st.write(f"Average score: {avg_display if avg_display is not None else 'n/a'}")
                rows = []
                for entry in last_run.get("conversations", []):
                    rows.append(
                        {
                            "conversation_id": entry.get("conversation_id"),
                            "average_score": entry.get("average_score"),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                for entry in last_run.get("conversations", []):
                    convo_id = entry.get("conversation_id")
                    with st.expander(f"Conversation {convo_id}", expanded=False):
                        st.write(f"Average score: {entry.get('average_score')}")
                        if entry.get("scores"):
                            st.json(entry.get("scores"))
                        if entry.get("rationales"):
                            st.json(entry.get("rationales"))
                        if entry.get("overall_notes"):
                            st.write(entry.get("overall_notes"))
                        with st.expander("Simulated transcript", expanded=False):
                            st.code(_format_transcript(entry.get("simulated_messages") or []), language="text")
