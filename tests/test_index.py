import numpy as np
import pytest

from footagefind.index import DIM, VideoIndex, sqlite_vec_available

from conftest import unit


def fill(ix, rng, n_frames=20, vid="v1"):
    ix.upsert_video(vid, f"/tmp/{vid}.avi", f"{vid}.avi", 10.0, n_frames / 2, 640, 480)
    vecs = []
    for i in range(n_frames):
        e = np.stack([unit(rng.normal(size=DIM)) for _ in range(3)])
        ix.add_items(vid, i * 0.5, i * 5, ["frame", "crop", "crop"], [None, "person", "car"],
                     [None, 0.9, 0.5], [None, (1, 2, 30, 40), (5, 5, 50, 50)], f"/thumbs/{i}.jpg", e)
        vecs.append(e)
    return np.concatenate(vecs)


def test_roundtrip_and_metadata(tmp_path, rng):
    ix = VideoIndex(tmp_path / "t.db")
    fill(ix, rng)
    assert ix.count() == 60 and ix.count("v1") == 60
    v = ix.videos()[0]
    assert v["name"] == "v1.avi" and v["fps"] == 10.0
    items = ix.get_items([1, 2])
    assert items[1].kind == "frame" and items[1].bbox is None
    assert items[2].kind == "crop" and items[2].label == "person" and items[2].bbox == (1, 2, 30, 40)
    meta = ix.meta_arrays()
    assert len(meta["id"]) == 60 and set(meta["kind"]) == {"frame", "crop"}


def test_similarity_is_exact_cosine_ranking(tmp_path, rng):
    ix = VideoIndex(tmp_path / "t.db")
    mat = fill(ix, rng)
    q = unit(rng.normal(size=DIM))
    ids, scores = ix.similarity(q)
    expected = np.sort(mat @ q)[::-1]
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)
    assert ids[0] == int(np.argmax(mat @ q)) + 1        # AUTOINCREMENT ids start at 1


def test_exact_match_ranks_first(tmp_path, rng):
    ix = VideoIndex(tmp_path / "t.db")
    mat = fill(ix, rng)
    ids, scores = ix.similarity(mat[17], k=3)
    assert ids[0] == 18 and scores[0] == pytest.approx(1.0, abs=1e-5)
    assert len(ids) == 3


def test_clear_video_and_reindex(tmp_path, rng):
    ix = VideoIndex(tmp_path / "t.db")
    fill(ix, rng, vid="a")
    fill(ix, rng, vid="b")
    ix.clear_video("a")
    assert ix.count("a") == 0 and ix.count("b") == 60
    ids, _ = ix.similarity(unit(rng.normal(size=DIM)))
    assert len(ids) == 60


def test_persists_across_connections(tmp_path, rng):
    p = tmp_path / "t.db"
    ix = VideoIndex(p)
    mat = fill(ix, rng)
    ix.close()
    ix2 = VideoIndex(p)
    ids, _ = ix2.similarity(mat[5], k=1)
    assert ids[0] == 6


def test_rejects_wrong_dimension(tmp_path):
    ix = VideoIndex(tmp_path / "t.db")
    with pytest.raises(AssertionError):
        ix.add_items("v", 0, 0, ["frame"], [None], [None], [None], None, np.zeros((1, 128), np.float32))


@pytest.mark.skipif(not sqlite_vec_available(), reason="sqlite-vec not installed")
def test_sqlite_vec_backend_matches_numpy(tmp_path, rng):
    a = VideoIndex(tmp_path / "a.db", backend="numpy")
    b = VideoIndex(tmp_path / "b.db", backend="sqlite-vec")
    mat = fill(a, np.random.default_rng(1))
    fill(b, np.random.default_rng(1))
    q = unit(rng.normal(size=DIM))
    ia, sa = a.similarity(q)
    ib, sb = b.similarity(q)
    np.testing.assert_array_equal(ia[:20], ib[:20])
    np.testing.assert_allclose(sa, sb, atol=1e-5)
    b.clear_video("v1")
    assert len(b.similarity(q)[0]) == 0
