"""Retrieval metrics for eval/run_eval.py (kept in the package so they are unit-tested)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def hit_is_correct(video_name: str, t: float, gt: list[dict], tol: float) -> bool:
    return any(g["video"] == video_name and g["start"] - tol <= t <= g["end"] + tol for g in gt)


def first_correct_rank(hits: Iterable, gt: list[dict], tol: float, name_of: dict[str, str]) -> int | None:
    """1-based rank of the first hit whose best timestamp falls in a GT range."""
    for r, h in enumerate(hits, 1):
        if hit_is_correct(name_of.get(h.video_id, h.video_id), h.t, gt, tol):
            return r
    return None


def summarize_ranks(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    n = len(ranks)
    vals = [math.inf if r is None else r for r in ranks]
    out = {f"recall@{k}": round(sum(v <= k for v in vals) / n, 4) if n else None for k in ks}
    med = float(np.median(vals)) if n else None
    out["median_rank"] = med if med is None or math.isfinite(med) else "inf"
    out["mrr"] = round(sum(0 if not math.isfinite(v) else 1 / v for v in vals) / n, 4) if n else None
    out["not_found"] = sum(r is None for r in ranks)
    out["n"] = n
    return out


def gt_coverage(timestamps: list[tuple[str, float]], gt: list[dict], tol: float) -> float:
    """Fraction of indexed (video, t) samples inside ground truth ~= random-ranking Recall@1."""
    if not timestamps:
        return 0.0
    return sum(hit_is_correct(v, t, gt, tol) for v, t in timestamps) / len(timestamps)
