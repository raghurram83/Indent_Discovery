from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir


@dataclass
class WorkspacePaths:
    root: Path

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def normalized_path(self) -> Path:
        return self.input_dir / "normalized_conversations.jsonl"

    @property
    def atoms_path(self) -> Path:
        return self.root / "stage_atoms" / "atoms.jsonl"

    @property
    def faq_catalogue_path(self) -> Path:
        return self.root / "stage_faq" / "faq_catalogue.json"

    @property
    def intent_flow_catalogue_path(self) -> Path:
        return self.root / "stage_catalogue" / "intent_flow_catalogue.json"

    @property
    def blueprint_path(self) -> Path:
        return self.root / "stage_blueprint" / "blueprint.json"

    @property
    def llm_cache_dir(self) -> Path:
        return self.root / "cache" / "llm"

    @property
    def embed_cache_dir(self) -> Path:
        return self.root / "cache" / "embed"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def get_workspace(root: Path, project_id: str) -> WorkspacePaths:
    ws = WorkspacePaths(root=root / project_id)
    ensure_dir(ws.root)
    ensure_dir(ws.logs_dir)
    return ws
