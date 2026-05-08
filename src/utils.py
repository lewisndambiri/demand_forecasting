from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)

def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_path(str(config_path))
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

