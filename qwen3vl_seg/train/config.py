"""YAML config loading with local ``includes`` support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    merged: dict[str, Any] = {}

    def load_one(current: Path) -> None:
        content = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        includes = content.pop("includes", []) or []
        for include in includes:
            include_path = (current.parent / include).resolve()
            load_one(include_path)
        merged.update(content)

    load_one(config_path)
    return merged
