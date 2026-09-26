"""Local vector index: SQLite for metadata + embeddings, numpy (or sqlite-vec) for search.

Everything lives in one ``.db`` file next to a folder of JPEG thumbnails, so a
shop owner can copy/delete an investigation as a unit. Brute-force cosine over
a few hundred thousand 512-d vectors is milliseconds with numpy, so an ANN
library is unnecessary at this scale. If the ``sqlite-vec`` extension is
installed it can be used instead (``backend="sqlite-vec"``); both backends
return identical rankings (see tests/test_index.py).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIM = 512

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    fps REAL, duration REAL, width INTEGER, height INTEGER,
    proxy_path TEXT,
    indexed_at REAL,
    stats_json TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id),
    t REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('frame', 'crop')),
    label TEXT, conf REAL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    thumb TEXT,
    emb BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS items_video_t ON items(video_id, t);
"""


@dataclass
class Item:
    id: int
    video_id: str
    t: float
    frame_idx: int
    kind: str
    label: str | None
    conf: float | None
    bbox: tuple[int, int, int, int] | None
    thumb: str | None


def sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401
        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.close()
        return True
    except Exception:
        return False


class VideoIndex:
    def __init__(self, db_path: str | Path, backend: str = "numpy"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.executescript(SCHEMA)
        self.backend = backend
        if backend == "sqlite-vec":
            import sqlite_vec
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{DIM}] distance_metric=cosine)"
            )
        elif backend != "numpy":
            raise ValueError(f"unknown index backend {backend!r}")
        self._cache: tuple[np.ndarray, np.ndarray] | None = None   # (ids, matrix)

    # ------------------------------------------------------------------ writes
    def upsert_video(self, vid: str, path: str, name: str, fps: float, duration: float,
                     width: int, height: int, proxy_path: str | None = None, stats: dict | None = None) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO videos(id, path, name, fps, duration, width, height, proxy_path, indexed_at, stats_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET path=excluded.path, name=excluded.name, "
                "fps=excluded.fps, duration=excluded.duration, width=excluded.width, height=excluded.height, "
                "proxy_path=COALESCE(excluded.proxy_path, videos.proxy_path), indexed_at=excluded.indexed_at, "
                "stats_json=COALESCE(excluded.stats_json, videos.stats_json)",
                (vid, path, name, fps, duration, width, height, proxy_path, time.time(),
                 json.dumps(stats) if stats is not None else None),
            )
            self.db.commit()

    def set_video_stats(self, vid: str, stats: dict) -> None:
        with self.lock:
            self.db.execute("UPDATE videos SET stats_json=? WHERE id=?", (json.dumps(stats), vid))
            self.db.commit()

    def clear_video(self, vid: str) -> None:
        with self.lock:
            if self.backend == "sqlite-vec":
                self.db.execute("DELETE FROM vec_items WHERE rowid IN (SELECT id FROM items WHERE video_id=?)", (vid,))
            self.db.execute("DELETE FROM items WHERE video_id=?", (vid,))
            self.db.commit()
            self._cache = None

    def add_items(self, video_id: str, t: float, frame_idx: int, kinds: list[str], labels: list[str | None],
                  confs: list[float | None], boxes: list[tuple | None], thumb: str | None,
                  embeddings: np.ndarray) -> list[int]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.ndim == 2 and embeddings.shape[1] == DIM, embeddings.shape
        ids = []
        with self.lock:
            for kind, label, conf, box, emb in zip(kinds, labels, confs, boxes, embeddings):
                b = box if box is not None else (None, None, None, None)
                cur = self.db.execute(
                    "INSERT INTO items(video_id, t, frame_idx, kind, label, conf, x1, y1, x2, y2, thumb, emb) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (video_id, float(t), int(frame_idx), kind, label, conf, *b, thumb, emb.tobytes()),
                )
                ids.append(cur.lastrowid)
                if self.backend == "sqlite-vec":
                    self.db.execute("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", (cur.lastrowid, emb.tobytes()))
            self.db.commit()
            self._cache = None
        return ids

    # ------------------------------------------------------------------- reads
    def videos(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, path, name, fps, duration, width, height, proxy_path, indexed_at, stats_json FROM videos "
            "ORDER BY indexed_at DESC"
        ).fetchall()
        keys = ["id", "path", "name", "fps", "duration", "width", "height", "proxy_path", "indexed_at", "stats"]
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["stats"] = json.loads(d["stats"]) if d["stats"] else None
            out.append(d)
        return out

    def video(self, vid: str) -> dict | None:
        return next((v for v in self.videos() if v["id"] == vid), None)

    def count(self, video_id: str | None = None) -> int:
        if video_id:
            return self.db.execute("SELECT COUNT(*) FROM items WHERE video_id=?", (video_id,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def get_items(self, ids) -> dict[int, Item]:
        ids = [int(i) for i in ids]
        out: dict[int, Item] = {}
        for s in range(0, len(ids), 900):
            chunk = ids[s : s + 900]
            q = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT id, video_id, t, frame_idx, kind, label, conf, x1, y1, x2, y2, thumb FROM items WHERE id IN ({q})",
                chunk,
            ):
                box = None if r[7] is None else (r[7], r[8], r[9], r[10])
                out[r[0]] = Item(r[0], r[1], r[2], r[3], r[4], r[5], r[6], box, r[11])
        return out

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if self._cache is None:
                rows = self.db.execute("SELECT id, emb FROM items ORDER BY id").fetchall()
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                mat = (np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(-1, DIM)
                       if rows else np.zeros((0, DIM), np.float32))
                self._cache = (ids, mat)
            return self._cache

    def meta_arrays(self) -> dict[str, np.ndarray]:
        """Per-item metadata aligned with the embedding matrix (for vectorised filtering)."""
        rows = self.db.execute("SELECT id, video_id, t, kind FROM items ORDER BY id").fetchall()
        return {
            "id": np.array([r[0] for r in rows], dtype=np.int64),
            "video_id": np.array([r[1] for r in rows], dtype=object),
            "t": np.array([r[2] for r in rows], dtype=np.float64),
            "kind": np.array([r[3] for r in rows], dtype=object),
        }

    def similarity(self, query: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (item_ids, cosine_scores) sorted by descending score."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        if self.backend == "sqlite-vec":
            n = self.count()
            k = min(k or n, n, 4096)   # vec0 caps k
            if k == 0:
                return np.zeros(0, np.int64), np.zeros(0, np.float32)
            with self.lock:
                rows = self.db.execute(
                    "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (q.tobytes(), k),
                ).fetchall()
            return (np.array([r[0] for r in rows], np.int64), 1.0 - np.array([r[1] for r in rows], np.float32))
        ids, mat = self._matrix()
        if len(ids) == 0:
            return ids, np.zeros(0, np.float32)
        scores = mat @ q
        order = np.argsort(-scores, kind="stable")
        if k:
            order = order[:k]
        return ids[order], scores[order]

    def all_embeddings(self, video_id: str | None = None, kind: str | None = None) -> np.ndarray:
        sql, args = "SELECT emb FROM items WHERE 1=1", []
        if video_id:
            sql += " AND video_id=?"; args.append(video_id)
        if kind:
            sql += " AND kind=?"; args.append(kind)
        rows = self.db.execute(sql + " ORDER BY id", args).fetchall()
        return np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(-1, DIM) if rows \
            else np.zeros((0, DIM), np.float32)

    def close(self) -> None:
        self.db.close()
