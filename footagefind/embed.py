"""CLIP image/text embedding on ONNX Runtime sessions (no PyTorch)."""
from __future__ import annotations

import cv2
import numpy as np

from .runtime import OrtModel
from .tokenizer import get_tokenizer

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def preprocess(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """open_clip eval transform: shortest-side resize (bicubic) -> centre crop -> normalise."""
    h, w = img_bgr.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    # INTER_AREA when shrinking approximates PIL's antialiased bicubic used by open_clip
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    img = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    top, left = (nh - size) // 2, (nw - size) // 2
    img = img[top : top + size, left : left + size, ::-1].astype(np.float32) / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1)


def square_crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    """Expand a detection box to a padded square *in the frame* before CLIP.

    Person boxes are tall and thin; CLIP's centre-crop would otherwise cut off
    heads/feet, and stretching would distort them. Taking a square window keeps
    aspect ratio and adds a little scene context (is the person near the gate?).
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = max(1, int(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad))))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    c, d = a + side, b + side
    crop = frame[max(0, b) : min(H, d), max(0, a) : min(W, c)]
    # pad with grey where the square runs off the frame edge
    top, left = max(0, -b), max(0, -a)
    bottom, right = max(0, d - H), max(0, c - W)
    if top or left or bottom or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(124, 116, 104))
    return crop


class ClipEmbedder:
    def __init__(self, image_model: OrtModel | None, text_model: OrtModel | None):
        self.image_model = image_model
        self.text_model = text_model
        self.tokenizer = get_tokenizer()

    def _run_batched(self, model: OrtModel, x: np.ndarray) -> np.ndarray:
        """Feed a static-batch model, padding the final partial batch."""
        bs = model.batch_size
        name = model.inputs[0].name
        outs = []
        for i in range(0, len(x), bs):
            chunk = x[i : i + bs]
            n = len(chunk)
            if n < bs:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], bs - n, axis=0)])
            outs.append(model.run({name: np.ascontiguousarray(chunk)})[0][:n])
        return np.concatenate(outs) if outs else np.zeros((0, 512), np.float32)

    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        if not images_bgr:
            return np.zeros((0, 512), np.float32)
        x = np.stack([preprocess(im) for im in images_bgr]).astype(np.float32)
        return l2norm(self._run_batched(self.image_model, x))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        toks = self.tokenizer(texts)
        dtype = self.text_model.inputs[0].type
        toks = toks.astype(np.int64 if "int64" in dtype else np.int32)
        return l2norm(self._run_batched(self.text_model, toks))
