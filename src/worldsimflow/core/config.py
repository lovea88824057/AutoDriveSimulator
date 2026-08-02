from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "runtime": {
        "steps": 120,
        "render_html": "outputs/demo_trace.html",
    },
    "policy": {
        "target_speed": 8.0,
    },
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _deep_merge(DEFAULT_CONFIG, data)


def resolve_project_path(root: Path, value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
