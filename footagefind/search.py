"""Text -> ranked, de-duplicated timestamps."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .index import VideoIndex


@dataclass
class Hit:
    video_id: str
    t: float                     # best-matching timestamp (where playback starts)
    score: float                 # CLIP cosine similarity of the best item
    t_start: float
    t_end: float
    item_id: int
    kind: str                    # 'frame' or 'crop'
    label: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    thumb: str | None = None
    frame_idx: int = 0
    merged: int = 1              # how many sampled timestamps were folded into this hit
    vlm_p_yes: float | None = None
    vlm_answer: str | None = None
    final_score: float | None = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def best_per_timestamp(ids: np.ndarray, scores: np.ndarray, meta: dict[str, np.ndarray],
                       kinds: tuple[str, ...] = ("frame", "crop"),
                       video_ids: list[str] | None = None) -> list[tuple[str, float, int, float]]:
    """Collapse items (full frame + crops) to one score per (video, t): the max.

    Returns [(video_id, t, item_id, score)] sorted by descending score.
    """
    pos = {int(i): k for k, i in enumerate(meta["id"])}
    best: dict[tuple[str, float], tuple[int, float]] = {}
    for iid, s in zip(ids.tolist(), scores.tolist()):
        k = pos.get(iid)
        if k is None:
            continue
        if meta["kind"][k] not in kinds:
            continue
        vid = meta["video_id"][k]
        if video_ids and vid not in video_ids:
            continue
        key = (vid, round(float(meta["t"][k]), 3))
        if key not in best or s > best[key][1]:
            best[key] = (iid, s)
    out = [(v, t, iid, s) for (v, t), (iid, s) in best.items()]
    out.sort(key=lambda r: -r[3])
    return out


def merge_hits(ranked: list[tuple[str, float, int, float]], merge_seconds: float,
               top_k: int | None = None) -> list[dict]:
    """Greedy temporal non-maximum suppression.

    Walk timestamps from best to worst score. A timestamp within
    ``merge_seconds`` of an already accepted hit's *anchor* (its best
    timestamp) in the same video is folded into that hit (widening its
    [t_start, t_end] range) instead of becoming a new result. Anchoring on the
    best timestamp (rather than the growing range) prevents a busy scene from
    chaining into one giant result.
    """
    hits: list[dict] = []
    for vid, t, iid, s in ranked:
        for h in hits:
            if h["video_id"] == vid and abs(t - h["t"]) <= merge_seconds:
                h["t_start"] = min(h["t_start"], t)
                h["t_end"] = max(h["t_end"], t)
                h["merged"] += 1
                break
        else:
            hits.append({"video_id": vid, "t": t, "item_id": iid, "score": s,
                         "t_start": t, "t_end": t, "merged": 1})
    hits.sort(key=lambda h: -h["score"])
    return hits[:top_k] if top_k else hits


def search_vector(index: VideoIndex, qvec: np.ndarray, top_k: int = 10, merge_seconds: float = 3.0,
                  use_crops: bool = True, video_ids: list[str] | None = None,
                  meta: dict | None = None) -> list[Hit]:
    ids, scores = index.similarity(qvec)
    meta = meta if meta is not None else index.meta_arrays()
    kinds = ("frame", "crop") if use_crops else ("frame",)
    ranked = best_per_timestamp(ids, scores, meta, kinds, video_ids)
    merged = merge_hits(ranked, merge_seconds, top_k)
    items = index.get_items([h["item_id"] for h in merged])
    out = []
    for h in merged:
        it = items[h["item_id"]]
        out.append(Hit(video_id=h["video_id"], t=h["t"], score=float(h["score"]), t_start=h["t_start"],
                       t_end=h["t_end"], item_id=h["item_id"], kind=it.kind, label=it.label, bbox=it.bbox,
                       thumb=it.thumb, frame_idx=it.frame_idx, merged=h["merged"]))
    return out
