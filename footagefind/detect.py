"""YOLOv8n object detection on an ONNX Runtime session (numpy pre/post-processing)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .runtime import OrtModel

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
}


@dataclass
class Detection:
    cls: int
    conf: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in original-frame pixels

    @property
    def label(self) -> str:
        return COCO_NAMES.get(self.cls, str(self.cls))


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def postprocess(raw: np.ndarray, ratio: float, pad: tuple[int, int], frame_shape: tuple[int, int],
                conf_threshold: float, iou_threshold: float, classes: list[int] | None,
                min_box_px: int = 0) -> list[Detection]:
    """raw: [84, N] YOLOv8 head output (cx, cy, w, h, 80 class scores)."""
    preds = raw.T
    scores_all = preds[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all[np.arange(len(cls)), cls]
    keep = conf >= conf_threshold
    if classes is not None:
        keep &= np.isin(cls, classes)
    if not keep.any():
        return []
    preds, cls, conf = preds[keep], cls[keep], conf[keep]
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / ratio
    y1 = (cy - bh / 2 - pad[1]) / ratio
    ww, hh = bw / ratio, bh / ratio
    idx = cv2.dnn.NMSBoxesBatched(
        np.stack([x1, y1, ww, hh], 1).tolist(), conf.tolist(), cls.tolist(), conf_threshold, iou_threshold
    )
    H, W = frame_shape
    out = []
    for i in np.array(idx).reshape(-1):
        a = int(max(0, x1[i])); b = int(max(0, y1[i]))
        c = int(min(W, x1[i] + ww[i])); d = int(min(H, y1[i] + hh[i]))
        if min(c - a, d - b) < min_box_px:
            continue
        out.append(Detection(int(cls[i]), float(conf[i]), (a, b, c, d)))
    out.sort(key=lambda det: -det.conf)
    return out


class Detector:
    def __init__(self, model: OrtModel, conf_threshold: float = 0.35, iou_threshold: float = 0.5,
                 classes: list[int] | None = None, min_box_px: int = 20, **_ignored):
        self.model = model
        self.size = int(model.inputs[0].shape[2])
        self.input_name = model.inputs[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.min_box_px = min_box_px

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        img, ratio, pad = letterbox(frame_bgr, self.size)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.model.run({self.input_name: np.ascontiguousarray(x)})[0][0]
        return postprocess(raw, ratio, pad, frame_bgr.shape[:2], self.conf_threshold,
                           self.iou_threshold, self.classes, self.min_box_px)
