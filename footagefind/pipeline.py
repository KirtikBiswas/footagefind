"""End-to-end engine: ingest -> motion gate -> detect -> embed -> index, and search.

Decode + motion gating run on a producer thread while detection/embedding run
on the calling thread, connected by a small bounded queue. OpenCV decoding and
ONNX Runtime inference both release the GIL, so on a laptop the CPU decodes
the next frames while the NPU (or other CPU cores) runs the models.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import audit
from .config import load_config, resolve
from .detect import Detector
from .embed import ClipEmbedder, square_crop
from .index import VideoIndex
from .ingest import make_browser_proxy, probe, sample_frames, video_id
from .motion import MotionGate
from .runtime import OrtModel, probe_node_placement
from .search import Hit, search_vector
from .verify import make_verifier, rerank

log = logging.getLogger("footagefind")

ProgressFn = Callable[[float, str], None]
_SENTINEL = object()


class FootageFind:
    def __init__(self, cfg: dict | None = None, db_path: str | Path | None = None, backend: str = "numpy"):
        self.cfg = cfg or load_config()
        self.db_path = resolve(db_path or self.cfg["index"]["db_path"])
        self.index = VideoIndex(self.db_path, backend=backend)
        self.thumbs_dir = resolve(self.cfg["index"]["thumbs_dir"])
        self._models: dict[str, OrtModel] = {}
        self._model_lock = threading.Lock()
        self._meta_cache: tuple[int, dict] | None = None

    # ------------------------------------------------------------ models (lazy)
    def model(self, key: str) -> OrtModel:
        with self._model_lock:
            if key not in self._models:
                self._models[key] = OrtModel(resolve(self.cfg["models"][key]), self.cfg["runtime"], name=key)
            return self._models[key]

    @property
    def embedder(self) -> ClipEmbedder:
        return ClipEmbedder(self.model("clip_image"), self.model("clip_text"))

    def text_embedder(self) -> ClipEmbedder:
        return ClipEmbedder(None, self.model("clip_text"))

    def detector(self) -> Detector:
        return Detector(self.model("yolo"), **self.cfg["detect"])

    def runtime_info(self, load_all: bool = False) -> list[dict]:
        if load_all:
            for k in ("clip_image", "clip_text", "yolo"):
                try:
                    self.model(k)
                except FileNotFoundError as e:
                    log.warning("%s", e)
        return [m.describe() for m in self._models.values()]

    def node_placement(self) -> list[dict]:
        return [probe_node_placement(resolve(self.cfg["models"][k]), self.cfg["runtime"])
                for k in ("clip_image", "clip_text", "yolo")]

    # ----------------------------------------------------------------- indexing
    def index_video(self, path: str | Path, progress: ProgressFn | None = None, use_crops: bool | None = None,
                    motion_enabled: bool | None = None, make_proxy: bool = True) -> dict:
        path = Path(path)
        icfg = self.cfg["index"]
        use_crops = icfg["use_crops"] if use_crops is None else use_crops
        mcfg = dict(self.cfg["motion"])
        if motion_enabled is not None:
            mcfg["enabled"] = motion_enabled
        info = probe(path)
        vid = video_id(path)
        progress = progress or (lambda f, m: None)
        t_start = time.perf_counter()

        embedder = ClipEmbedder(self.model("clip_image"), None)
        detector = self.detector() if use_crops else None
        gate = MotionGate(**mcfg)
        sample_fps = float(icfg["sample_fps"])
        expected = max(1, int(info.duration * sample_fps))
        self.index.upsert_video(vid, str(path.resolve()), path.name, info.fps, info.duration, info.width, info.height)
        self.index.clear_video(vid)
        thumb_dir = self.thumbs_dir / vid
        thumb_dir.mkdir(parents=True, exist_ok=True)

        timings = {"decode_gate_s": 0.0, "detect_s": 0.0, "embed_s": 0.0, "write_s": 0.0}
        q: queue.Queue = queue.Queue(maxsize=8)
        err: list[BaseException] = []

        def producer() -> None:
            try:
                t0 = time.perf_counter()
                for sf in sample_frames(path, sample_fps):
                    keep = gate.keep(sf.image, sf.t)
                    timings["decode_gate_s"] += time.perf_counter() - t0
                    q.put((sf, keep))
                    t0 = time.perf_counter()
            except BaseException as e:  # surfaced on the consumer thread
                err.append(e)
            finally:
                q.put(_SENTINEL)

        th = threading.Thread(target=producer, name="ff-decode", daemon=True)
        th.start()
        n_items = n_crops = n_frames = 0
        seen = 0
        while True:
            obj = q.get()
            if obj is _SENTINEL:
                break
            sf, keep = obj
            seen += 1
            if seen % 5 == 0:
                progress(min(0.99, seen / expected),
                         f"t={sf.t:6.1f}s  kept {gate.stats.kept}/{gate.stats.seen} frames, {n_crops} crops")
            if not keep:
                continue
            n_frames += 1
            frame = sf.image
            images, kinds, labels, confs, boxes = [frame], ["frame"], [None], [None], [None]
            if detector is not None:
                t0 = time.perf_counter()
                dets = detector(frame)[: int(self.cfg["detect"]["max_crops_per_frame"])]
                timings["detect_s"] += time.perf_counter() - t0
                for d in dets:
                    images.append(square_crop(frame, d.box))
                    kinds.append("crop"); labels.append(d.label); confs.append(d.conf); boxes.append(d.box)
                n_crops += len(dets)
            t0 = time.perf_counter()
            embs = embedder.embed_images(images)
            timings["embed_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            tw = int(icfg["thumb_width"])
            thumb = thumb_dir / f"{sf.frame_idx:07d}.jpg"
            h, w = frame.shape[:2]
            cv2.imwrite(str(thumb), cv2.resize(frame, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.index.add_items(vid, sf.t, sf.frame_idx, kinds, labels, confs, boxes, str(thumb), embs)
            timings["write_s"] += time.perf_counter() - t0
            n_items += len(images)
        th.join()
        if err:
            raise err[0]
        wall = time.perf_counter() - t_start

        proxy = None
        if make_proxy:
            progress(0.995, "creating browser-playable copy")
            try:
                proxy = str(make_browser_proxy(path, resolve(icfg["proxy_dir"]) / f"{vid}.webm"))
            except Exception as e:  # playback is a convenience, never fatal
                log.warning("proxy creation failed: %s", e)
        stats = {
            "video_id": vid, "video": path.name, "duration_s": round(info.duration, 2),
            "sample_fps": sample_fps, "use_crops": use_crops, "motion": gate.stats.as_dict(),
            "motion_method": mcfg["method"] if mcfg["enabled"] else "off",
            "frames_embedded": n_frames, "crops_embedded": n_crops, "items": n_items,
            "index_wall_s": round(wall, 2),
            "sampled_frames_per_s": round(gate.stats.seen / wall, 2) if wall else None,
            "realtime_factor": round(info.duration / wall, 2) if wall else None,
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "providers": {k: m.primary_provider for k, m in self._models.items()},
        }
        self.index.upsert_video(vid, str(path.resolve()), path.name, info.fps, info.duration, info.width,
                                info.height, proxy_path=proxy, stats=stats)
        self._meta_cache = None
        log.info("indexed %s: %s", path.name, stats)
        progress(1.0, "done")
        return stats

    # ------------------------------------------------------------------- search
    def _meta(self) -> dict:
        n = self.index.count()
        if self._meta_cache is None or self._meta_cache[0] != n:
            self._meta_cache = (n, self.index.meta_arrays())
        return self._meta_cache[1]

    def encode_query(self, query: str) -> np.ndarray:
        return self.text_embedder().embed_texts([query])[0]

    def search(self, query: str, top_k: int | None = None, video_ids: list[str] | None = None,
               use_crops: bool = True, verify: bool = False, audit_log: bool = True) -> list[Hit]:
        scfg = self.cfg["search"]
        top_k = top_k or int(scfg["top_k"])
        qvec = self.encode_query(query)
        hits = search_vector(self.index, qvec, top_k=top_k, merge_seconds=float(scfg["merge_seconds"]),
                             use_crops=use_crops, video_ids=video_ids, meta=self._meta())
        if verify:
            vcfg = dict(self.cfg["verify"])
            verifier = make_verifier(vcfg, enabled=True)
            images = [self.frame_at(h.video_id, h.frame_idx) for h in hits]
            hits = rerank(hits, images, query, verifier, weight=float(vcfg.get("weight", 0.5)))
        if audit_log:
            names = [v["name"] for v in self.index.videos() if not video_ids or v["id"] in video_ids]
            audit.log_query(resolve(self.cfg["audit"]["log_path"]), query, names, len(hits), verify=verify)
        return hits

    def frame_at(self, vid: str, frame_idx: int) -> np.ndarray:
        v = self.index.video(vid)
        cap = cv2.VideoCapture(v["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"cannot read frame {frame_idx} of {v['path']}")
        return img
