import numpy as np

from footagefind.index import DIM, VideoIndex
from footagefind.search import best_per_timestamp, merge_hits, search_vector

from conftest import unit


def test_merge_folds_neighbours_into_best_hit():
    ranked = [("v", 10.0, 1, 0.9), ("v", 10.5, 2, 0.85), ("v", 11.0, 3, 0.8), ("v", 30.0, 4, 0.7),
              ("v", 9.0, 5, 0.6), ("w", 10.0, 6, 0.5)]
    hits = merge_hits(ranked, merge_seconds=2.0)
    assert [(h["video_id"], h["t"]) for h in hits] == [("v", 10.0), ("v", 30.0), ("w", 10.0)]
    assert hits[0]["t_start"] == 9.0 and hits[0]["t_end"] == 11.0 and hits[0]["merged"] == 4


def test_merge_is_anchored_not_chained():
    """A run of frames 1 s apart must not collapse into one giant hit."""
    ranked = [("v", float(t), t, 1.0 - t * 0.01) for t in range(0, 20)]
    hits = merge_hits(ranked, merge_seconds=3.0)
    assert len(hits) > 1
    for h in hits:
        assert h["t_end"] - h["t"] <= 3.0 and h["t"] - h["t_start"] <= 3.0


def test_merge_zero_window_and_top_k():
    ranked = [("v", float(t), t, 1.0 - t * 0.01) for t in range(10)]
    assert len(merge_hits(ranked, 0.0)) == 10
    assert len(merge_hits(ranked, 0.0, top_k=3)) == 3


def test_merge_keeps_videos_separate():
    ranked = [("a", 5.0, 1, 0.9), ("b", 5.0, 2, 0.8), ("a", 5.5, 3, 0.7)]
    hits = merge_hits(ranked, 3.0)
    assert len(hits) == 2 and hits[0]["merged"] == 2


def test_best_per_timestamp_takes_max_over_frame_and_crops():
    meta = {"id": np.array([1, 2, 3, 4]), "video_id": np.array(["v", "v", "v", "v"], dtype=object),
            "t": np.array([0.0, 0.0, 0.5, 0.5]), "kind": np.array(["frame", "crop", "frame", "crop"], dtype=object)}
    ids, scores = np.array([2, 3, 1, 4]), np.array([0.9, 0.5, 0.4, 0.3])
    out = best_per_timestamp(ids, scores, meta)
    assert out == [("v", 0.0, 2, 0.9), ("v", 0.5, 3, 0.5)]
    frames_only = best_per_timestamp(ids, scores, meta, kinds=("frame",))
    assert frames_only == [("v", 0.5, 3, 0.5), ("v", 0.0, 1, 0.4)]
    assert best_per_timestamp(ids, scores, meta, video_ids=["other"]) == []


def test_search_vector_end_to_end(tmp_path, rng):
    ix = VideoIndex(tmp_path / "s.db")
    ix.upsert_video("v", "/v.avi", "v.avi", 10, 20, 64, 64)
    target = unit(rng.normal(size=DIM))
    for i in range(40):                               # 0..19.5 s at 2 fps
        t = i * 0.5
        near = 7.0 <= t <= 8.0                        # the event
        frame = unit(target + rng.normal(scale=3.0, size=DIM))
        crop = unit(target + rng.normal(scale=0.3 if near else 5.0, size=DIM))
        ix.add_items("v", t, i * 5, ["frame", "crop"], [None, "person"], [None, 0.9],
                     [None, (1, 1, 10, 10)], None, np.stack([frame, crop]))
    hits = search_vector(ix, target, top_k=5, merge_seconds=2.0)
    assert 7.0 <= hits[0].t <= 8.0 and hits[0].kind == "crop" and hits[0].label == "person"
    assert hits[0].t_start >= 5.0 and hits[0].t_end <= 10.0
    assert all(abs(h.t - hits[0].t) > 2.0 for h in hits[1:])
    # frames-only search ignores the crop embeddings entirely
    fo = search_vector(ix, target, top_k=5, merge_seconds=2.0, use_crops=False)
    assert all(h.kind == "frame" for h in fo)
