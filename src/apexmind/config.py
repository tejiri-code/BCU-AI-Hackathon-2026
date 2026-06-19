"""Load the YAML config into a typed object and resolve repo-relative paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    # src/apexmind/config.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: Dict[str, Any]
    root: Path

    @property
    def team_name(self) -> str:
        return self.raw.get("team_name", "ApexMind")

    def path(self, key: str) -> Path:
        """Resolve a path under paths.<key> relative to the repo root."""
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    def model(self, name: str) -> Dict[str, Any]:
        return self.raw["models"][name]

    def model_path(self, name: str) -> Path:
        return self.path("models_dir") / self.model(name)["file"]

    @property
    def llama(self) -> Dict[str, Any]:
        return self.raw["llama"]

    @property
    def generation(self) -> Dict[str, Any]:
        return self.raw["generation"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})


def load_config(path: str | Path | None = None) -> Config:
    root = repo_root()
    cfg_path = Path(path) if path else (root / "config" / "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)
