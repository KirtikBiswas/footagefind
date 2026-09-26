"""Configuration loading.

``configs/default.toml`` is always loaded first; an optional second TOML file
(e.g. ``configs/snapdragon.toml``) is deep-merged on top, followed by any
programmatic overrides.
"""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    path = path or os.environ.get("FOOTAGEFIND_CONFIG")
    if path:
        with open(path, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
