from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .utils import ensure_dir, sha256_text


class FileCache:
    def __init__(self, root: Path, log_path: Path) -> None:
        self.root = root
        ensure_dir(self.root)
        self.logger = logging.getLogger(f"cache:{root}")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def key_for(self, model: str, prompt: str, payload: str) -> str:
        return sha256_text(f"{model}\n{prompt}\n{payload}")

    def get(self, key: str) -> Optional[Any]:
        path = self.root / f"{key}.json"
        if not path.exists():
            self.logger.info("cache_miss %s", key)
            return None
        self.logger.info("cache_hit %s", key)
        return json.loads(path.read_text())

    def set(self, key: str, value: Any) -> None:
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=True, indent=2))
        self.logger.info("cache_write %s", key)
