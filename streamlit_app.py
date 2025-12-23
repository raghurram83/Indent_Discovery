from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from openai import OpenAI
from shutil import rmtree

from src.atoms import SYSTEM_PROMPT as ATOMS_SYSTEM_PROMPT
from src.atoms import USER_TEMPLATE as ATOMS_USER_TEMPLATE
from src.atoms import extract_atoms
from src.blueprint import build_blueprint
from src.cache import FileCache
from src.faq import SYSTEM_PROMPT as FAQ_SYSTEM_PROMPT
from src.faq import USER_TEMPLATE as FAQ_USER_TEMPLATE
from src.faq import build_faq_catalogue
from src.importer import normalize_records, read_raw_records
from src.intent_flow import SYSTEM_PROMPT as INTENT_SYSTEM_PROMPT
from src.intent_flow import USER_TEMPLATE as INTENT_USER_TEMPLATE
from src.intent_flow import build_intent_flow_catalogue
from src.storage import get_workspace
from src.utils import load_json, load_jsonl, save_json, save_jsonl


APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT / "workspace"

st.set_page_config(page_title="Conversation-Driven Intent/FAQ/Blueprint", layout="wide")


def list_projects() -> List[str]:
    if not WORKSPACE_ROOT.exists():
        return []
    return sorted([p.name for p in WORKSPACE_ROOT.iterdir() if p.is_dir()])


def get_openai_client(api_key: str) -> OpenAI | None:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


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

def load_prompts(ws_root: Path) -> Dict[str, str]:
    path = ws_root / "prompts.json"
    if path.exists():
        return load_json(path)
    return {}


def save_prompts(ws_root: Path, prompts: Dict[str, str]) -> None:
    save_json(ws_root / "prompts.json", prompts)


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


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        rmtree(path)
    else:
        path.unlink()


st.title("Conversation-Driven Intent/FAQ/Blueprint Extraction")

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
</style>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Project")
    existing = list_projects()
    selected = st.selectbox("Select project", options=[""] + existing, format_func=lambda x: x or "Choose...")
    new_project = st.text_input("Or create new project id", value="indent_discovery_v2")
    project_id = selected or new_project
    st.write(f"Active project: **{project_id}**")

    ws = get_workspace(WORKSPACE_ROOT, project_id)
    run_state = load_run_state(ws.root)
    prompts = load_prompts(ws.root)

    st.divider()
    st.header("Upload")
    upload_file = st.file_uploader("Upload conversations (.json or .jsonl)", type=["json", "jsonl"])
    if upload_file is not None:
        raw_path = ws.input_dir / upload_file.name
        ws.input_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(upload_file.getvalue())
        st.success(f"Saved to {raw_path.name}")

    st.divider()
    st.header("Settings")
    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    embed_model = st.text_input("Embedding model", value="text-embedding-3-small")
    threshold = st.slider("Similarity threshold", 0.5, 0.95, 0.82, 0.01)
    min_faq_freq = st.number_input("Min FAQ frequency", min_value=1, value=3, step=1)
    min_intent_freq = st.number_input("Min intent frequency", min_value=1, value=3, step=1)
    redact_pii = st.checkbox("Redact PII", value=False)
    openai_key = st.text_input("OPENAI_API_KEY", value="", type="password")

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

    run_blueprint = st.button("Build Blueprint")
    st.divider()
    st.subheader("Clear outputs")
    clear_all_outputs = st.button("Clear all outputs")
    clear_normalize = st.button("Clear normalization output")
    clear_atoms = st.button("Clear atoms output")
    clear_faqs = st.button("Clear FAQs output")
    clear_intents = st.button("Clear intents/flows output")
    clear_blueprint = st.button("Clear blueprint output")

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

if clear_all_outputs:
    _remove_path(ws.normalized_path)
    _remove_path(ws.atoms_path)
    _remove_path(ws.faq_catalogue_path)
    _remove_path(ws.intent_flow_catalogue_path)
    _remove_path(ws.blueprint_path)
    _remove_path(ws.root / "logs")
    _remove_path(ws.root / "run_state.json")
    st.success("Cleared all outputs and run state.")

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
    raw_files = list(ws.input_dir.glob("*.json")) + list(ws.input_dir.glob("*.jsonl"))
    if not raw_files:
        st.warning("Upload a JSON/JSONL file first.")
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
        if client is None:
            st.warning("OPENAI_API_KEY missing; using fallback FAQ generation.")
        faq_system = get_prompt(prompts, "faq.system", FAQ_SYSTEM_PROMPT)
        faq_user = get_prompt(prompts, "faq.user", FAQ_USER_TEMPLATE)
        faq_entries = build_faq_catalogue(
            atoms_rows,
            client,
            llm_model,
            llm_cache,
            embed_model,
            embed_cache,
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
        if client is None:
            st.warning("OPENAI_API_KEY missing; using fallback intent generation.")
        intent_system = get_prompt(prompts, "intent_flow.system", INTENT_SYSTEM_PROMPT)
        intent_user = get_prompt(prompts, "intent_flow.user", INTENT_USER_TEMPLATE)
        intents = build_intent_flow_catalogue(
            atoms_rows,
            client,
            llm_model,
            llm_cache,
            embed_model,
            embed_cache,
            system_prompt=intent_system,
            user_template=intent_user,
            threshold=threshold,
            min_freq=min_intent_freq,
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
        blueprint = build_blueprint(atoms_rows, faq_entries)
        save_json(ws.blueprint_path, blueprint)
        mark_stage(run_state, "blueprint")
        save_run_state(ws.root, run_state)
        st.success("Saved blueprint.")
    else:
        st.warning("Run atoms extraction and FAQs first.")


# ---------- Main UI tabs ----------

tabs = st.tabs(
    [
        "Preview",
        "Atoms",
        "FAQs",
        "Intents/Flows",
        "Blueprint",
        "Export",
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
        summary = [
            {
                "conversation_id": a.get("conversation_id"),
                "faq_candidates": len(a.get("faq_candidates", []) or []),
                "flow_candidates": len(a.get("flow_candidates", []) or []),
                "policy": ", ".join((a.get("blueprint_signals", {}) or {}).get("policy", [])[:3]),
                "tone": ", ".join((a.get("blueprint_signals", {}) or {}).get("tone", [])[:3]),
            }
            for a in atoms_rows
        ]
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        convo_ids = [a.get("conversation_id") for a in atoms_rows]
        selected = st.selectbox("Atoms detail", convo_ids, key="atoms_detail")
        detail = next((a for a in atoms_rows if a.get("conversation_id") == selected), None)
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
            if "intent_open" not in st.session_state or st.session_state["intent_open"] not in intent_names:
                st.session_state["intent_open"] = intent_names[0]
            left, right = st.columns([2, 5], gap="large")
            with left:
                st.markdown("**Intent list**")
                selected = st.radio(" ", intent_names, key="intent_detail")
                st.session_state["intent_open"] = selected
            selected_intent = next((i for i in filtered_intents if i.get("intent_name") == selected), None)
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
                        scenario_labels = []
                        scenario_by_label = {}
                        for idx, scenario in enumerate(scenarios, start=1):
                            scenario_confidence = scenario.get("scenario_confidence") or {}
                            automation = scenario.get("automation_suitability") or {}
                            label = (
                                f"{idx}. {scenario.get('scenario_name')} | "
                                f"freq {scenario.get('frequency')} | "
                                f"conf {scenario_confidence.get('score_0_to_1')} | "
                                f"auto {automation.get('score_0_to_1')}"
                            )
                            scenario_labels.append(label)
                            scenario_by_label[label] = scenario

                        scenario_cols = st.columns([2, 5], gap="large")
                        with scenario_cols[0]:
                            st.markdown("**Scenarios**")
                            selected_label = st.radio(
                                " ",
                                scenario_labels,
                                key=f"scenario_select_{intent.get('intent_name')}",
                            )
                        scenario = scenario_by_label.get(selected_label)
                        with scenario_cols[1]:
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
                    st.markdown("**Intent JSON**")
                    st.json(intent)
    else:
        st.info("Run intent/flow stage.")

with tabs[4]:
    st.subheader("Blueprint")
    if ws.blueprint_path.exists():
        blueprint = load_json(ws.blueprint_path)
        st.json(blueprint)
    else:
        st.info("Run blueprint stage.")

with tabs[5]:
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
