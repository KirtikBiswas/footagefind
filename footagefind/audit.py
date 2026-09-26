"""Append-only local audit log of every search (JSON Lines).

Searching CCTV is a sensitive act. Every query is recorded locally with a
timestamp and the videos searched, so a housing-society committee or shop
owner can later review who looked for what. Nothing leaves the machine.
"""
from __future__ import annotations

import datetime as dt
import getpass
import json
import threading
from pathlib import Path

_lock = threading.Lock()


def log_query(log_path: str | Path, query: str, videos: list[str], n_results: int,
              verify: bool = False, extra: dict | None = None) -> dict:
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover
        user = "unknown"
    rec = {
        "time": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "os_user": user,
        "query": query,
        "videos": videos,
        "n_results": n_results,
        "verify": verify,
    }
    if extra:
        rec.update(extra)
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_log(log_path: str | Path, last: int | None = None) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    if last:
        lines = lines[-last:]
    return [json.loads(line) for line in lines if line.strip()]
