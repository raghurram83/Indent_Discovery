from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..utils import ensure_dir, now_ts, sha256_text, load_json, save_json
from .client import ParlantClient, ParlantError
from .mappers import map_global_guidelines, map_journey_guidelines, map_journeys, map_tools


def _log(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_ts()}] {message}\n")


def _hash_file(path: Path) -> str:
    return sha256_text(path.read_text())


def _load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {}


def sync_project_to_parlant(project_path: Path, base_url: str, dry_run: bool = False) -> Dict[str, Any]:
    project_id = project_path.name
    log_path = project_path / "parlant_sync.log"
    manifest_path = project_path / "parlant_sync_manifest.json"
    result: Dict[str, Any] = {
        "ok": False,
        "project_id": project_id,
        "journeys_upserted": 0,
        "guidelines_upserted": 0,
        "tools_registered": 0,
        "skipped": False,
        "errors": [],
    }

    if not base_url:
        result["errors"].append("PARLANT_BASE_URL is required.")
        _log(log_path, "Missing PARLANT_BASE_URL; sync aborted.")
        return result

    intent_path = project_path / "stage_catalogue" / "intent_flow_catalogue.json"
    blueprint_path = project_path / "stage_blueprint" / "blueprint.json"

    if not intent_path.exists():
        result["errors"].append("Intent/flow catalogue not found.")
    if not blueprint_path.exists():
        result["errors"].append("Blueprint not found.")
    if result["errors"]:
        _log(log_path, "Missing required artifacts; sync aborted.")
        return result

    manifest = _load_manifest(manifest_path)
    intent_hash = _hash_file(intent_path)
    blueprint_hash = _hash_file(blueprint_path)
    if (
        manifest.get("intent_flow_hash") == intent_hash
        and manifest.get("blueprint_hash") == blueprint_hash
        and not dry_run
    ):
        result["ok"] = True
        result["skipped"] = True
        _log(log_path, "No changes detected; sync skipped.")
        return result

    intent_payload = load_json(intent_path)
    blueprint_payload = load_json(blueprint_path)

    journeys = map_journeys(intent_payload, project_id)
    global_guidelines = map_global_guidelines(blueprint_payload)
    journey_guidelines = map_journey_guidelines(journeys, blueprint_payload)
    tools = map_tools(journeys)

    if dry_run:
        result.update(
            {
                "ok": True,
                "journeys_upserted": len(journeys),
                "guidelines_upserted": len(global_guidelines) + len(journey_guidelines),
                "tools_registered": len(tools),
            }
        )
        _log(log_path, "Dry run complete; no changes pushed.")
        return result

    client = ParlantClient(base_url)
    auto_info = client.auto_configure()
    if auto_info.get("ok"):
        _log(log_path, f"Auto-configured Parlant endpoints from {auto_info.get('source')}: {auto_info.get('updates')}")
    elif auto_info.get("paths") is not None:
        _log(
            log_path,
            "Parlant OpenAPI is missing journey/guideline endpoints. "
            "Available paths: " + ", ".join(auto_info.get("paths") or []),
        )
        result["errors"].append(
            "Parlant server does not expose journeys/guidelines/sessions. "
            "Enable those modules and restart the server."
        )
        return result
    try:
        for tool in tools:
            client.upsert_tool(tool)
            result["tools_registered"] += 1
        for guideline in global_guidelines + journey_guidelines:
            client.upsert_guideline(guideline)
            result["guidelines_upserted"] += 1
        for journey in journeys:
            client.upsert_journey(journey)
            result["journeys_upserted"] += 1
    except ParlantError as exc:
        result["errors"].append(str(exc))
        _log(log_path, f"Sync failed: {exc}")
        return result

    manifest_payload = {
        "project_id": project_id,
        "intent_flow_hash": intent_hash,
        "blueprint_hash": blueprint_hash,
        "synced_at": now_ts(),
        "journeys": len(journeys),
        "guidelines": len(global_guidelines) + len(journey_guidelines),
        "tools": len(tools),
        "base_url": base_url,
    }
    save_json(manifest_path, manifest_payload)
    result["ok"] = True
    _log(log_path, f"Sync complete: {manifest_payload}")
    return result
