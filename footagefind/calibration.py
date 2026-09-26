"""Calibration data for INT8 / w8a16 quantization, drawn from an indexed video.

Used by ``scripts/quantize_onnx.py`` (ONNX Runtime static QDQ on CPU) and by
``scripts/aihub_compile_profile.py`` (Qualcomm AI Hub quantize jobs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .detect import letterbox
from .embed import preprocess, square_crop
from .ingest import sample_frames
from .tokenizer import get_tokenizer

# Generic CCTV-style captions for text-encoder calibration. Deliberately
# disjoint from eval/queries.json so quantization never sees the test queries.
CALIBRATION_CAPTIONS = [
    "a person walking", "a man walking on a pavement", "a woman walking down a street",
    "two people talking", "a crowd outside a shop", "a delivery rider on a scooter",
    "a car parked outside a gate", "a motorcycle entering a building compound",
    "an auto rickshaw on the road", "a bicycle leaning against a wall", "a truck unloading boxes",
    "a person carrying a backpack", "a woman carrying a handbag", "a man with a suitcase",
    "someone holding an umbrella", "a child running", "a security guard at the entrance",
    "a person climbing a wall", "a man opening a shutter", "a shop counter with a cashier",
    "an empty corridor", "an empty parking lot at night", "a staircase in an apartment building",
    "a lift lobby", "people waiting at a bus stop", "a man wearing a helmet", "a woman in a saree",
    "a man in a white kurta", "a person in a yellow raincoat", "a person wearing a cap",
    "a person in a green t-shirt", "someone in a black hoodie", "a person with a mask on their face",
    "a dog on the street", "a cow on the road", "a cyclist on the street", "a scooter parked by a tree",
    "a gate opening", "a person sitting on a bench", "a man smoking near a wall",
    "a person looking into a car window", "a group of students", "an old man with a walking stick",
    "a person on a phone call", "a courier handing over a parcel", "a person pushing a cart",
    "a van driving away", "a person jumping over a fence", "people entering an office",
    "a person kneeling on the ground", "a cluttered storeroom", "a person at an ATM",
    "a man standing still", "a woman waving", "two men shaking hands", "a person picking something up",
    "a street at dusk", "a rainy street", "a person running across the road", "trees and a lamp post",
]


def calibration_frames(video: str | Path, n: int = 200, sample_fps: float = 2.0,
                       detector=None, max_crops_per_frame: int = 3) -> list[np.ndarray]:
    """~n BGR images (frames plus, if a detector is given, some detection crops)."""
    frames = [sf.image for sf in sample_frames(video, sample_fps)]
    if not frames:
        raise ValueError(f"no frames decoded from {video}")
    images: list[np.ndarray] = []
    step = max(1, len(frames) // n)
    for f in frames[::step]:
        images.append(f)
        if detector is not None:
            for d in detector(f)[:max_crops_per_frame]:
                images.append(square_crop(f, d.box))
    rng = np.random.default_rng(0)
    if len(images) > n:
        images = [images[i] for i in sorted(rng.choice(len(images), n, replace=False))]
    return images


def clip_image_calibration(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([preprocess(im) for im in images]).astype(np.float32)[:, None]  # [N,1,3,224,224]


def yolo_calibration(frames: list[np.ndarray], size: int = 640) -> np.ndarray:
    out = []
    for f in frames:
        img, _, _ = letterbox(f, size)
        out.append(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
    return np.stack(out)[:, None]  # [N,1,3,640,640]


def clip_text_calibration() -> np.ndarray:
    return get_tokenizer()(CALIBRATION_CAPTIONS)[:, None]  # [N,1,77] int32
