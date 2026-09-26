"""Video decoding and fixed-rate frame sampling (OpenCV)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class SampledFrame:
    frame_idx: int
    t: float            # seconds from start of file
    image: np.ndarray   # BGR uint8


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=str(path),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def video_id(path: str | Path) -> str:
    """Stable id from file name + size + first 1 MB (cheap, survives moves)."""
    p = Path(path)
    h = hashlib.sha1(p.name.encode())
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:12]


def sample_frames(path: str | Path, sample_fps: float = 2.0) -> Iterator[SampledFrame]:
    """Decode sequentially and yield frames at ~``sample_fps``.

    Sequential ``grab()`` is used instead of seeking because seeking in DVR
    exports (often badly muxed AVI/H.264) is unreliable and slow.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    step = max(fps / max(sample_fps, 1e-6), 1.0)
    next_pick = 0.0
    idx = 0
    try:
        while cap.grab():
            if idx + 1e-6 >= next_pick:
                ok, img = cap.retrieve()
                if ok:
                    yield SampledFrame(idx, idx / fps, img)
                next_pick += step
            idx += 1
    finally:
        cap.release()


def make_browser_proxy(src: str | Path, dst: str | Path, max_width: int = 960) -> Path:
    """Transcode to VP8 WebM (plays in Chrome/Edge, no ffmpeg dependency)."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
    tmp = dst.with_suffix(".tmp.webm")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV build lacks a VP8 encoder; cannot create browser proxy")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(img, size) if scale < 1.0 else img)
    writer.release()
    cap.release()
    tmp.replace(dst)
    return dst
