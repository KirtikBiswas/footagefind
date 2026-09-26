"""Export CLIP image encoder, CLIP text encoder and YOLOv8n to static-shape ONNX.

Static shapes are required for the Qualcomm QNN HTP backend (and AI Hub
compile jobs), so batch size, image size and token length are fixed at export.

    python scripts/export_onnx.py                 # batch 1 (NPU-friendly default)
    python scripts/export_onnx.py --image-batch 8 # bigger static batch for CPU throughput

This is the only place PyTorch is used. Outputs go to models/onnx/ together
with export_report.json (ORT vs PyTorch max-abs-diff for each model).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIP_ARCH = "ViT-B-32-quickgelu"
CLIP_CKPT = ROOT / "models" / "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
YOLO_CKPT = ROOT / "models" / "yolov8n.pt"
OUT = ROOT / "models" / "onnx"
OPSET = 17


def export_clip(image_batch: int, text_batch: int, report: dict) -> None:
    import onnxruntime as ort
    import open_clip
    import torch
    import torch.nn.functional as F

    from footagefind.tokenizer import get_tokenizer

    model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=str(CLIP_CKPT))
    model.eval()
    # The fused nn.MultiheadAttention fast path (aten::_native_multi_head_attention)
    # has no ONNX symbolic; the plain path exports to standard MatMul/Softmax ops.
    torch.backends.mha.set_fastpath_enabled(False)

    class ImageEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            return F.normalize(self.m.encode_image(image), dim=-1)

    class TextEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):  # int32 in: friendlier to NPUs than int64
            return F.normalize(self.m.encode_text(tokens.long()), dim=-1)

    OUT.mkdir(parents=True, exist_ok=True)
    img = torch.randn(image_batch, 3, 224, 224)
    toks = torch.from_numpy(get_tokenizer()(["a person carrying a large bag near the gate"] * text_batch))

    for name, module, example, in_name in [
        ("clip_image", ImageEncoder(model), img, "image"),
        ("clip_text", TextEncoder(model), toks, "tokens"),
    ]:
        path = OUT / f"{name}.onnx"
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(
                module, (example,), str(path), input_names=[in_name], output_names=["embedding"],
                opset_version=OPSET, dynamic_axes=None, dynamo=False, do_constant_folding=True,
            )
            ref = module(example).numpy()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {in_name: example.numpy()})[0]
        report[name] = {
            "file": path.name, "input": in_name, "input_shape": list(example.shape),
            "input_dtype": str(example.numpy().dtype), "output_shape": list(got.shape),
            "max_abs_diff_vs_torch": float(np.abs(ref - got).max()),
            "export_seconds": round(time.time() - t0, 1), "size_mb": round(path.stat().st_size / 1e6, 1),
        }
        print(name, report[name])


def export_yolo(report: dict) -> None:
    import onnxruntime as ort
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(YOLO_CKPT))
    produced = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=OPSET, batch=1))
    path = OUT / "yolov8n.onnx"
    shutil.move(str(produced), path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    report["yolov8n"] = {
        "file": path.name, "input": inp.name, "input_shape": inp.shape,
        "output_shape": sess.get_outputs()[0].shape,
        "export_seconds": round(time.time() - t0, 1), "size_mb": round(path.stat().st_size / 1e6, 1),
    }
    print("yolov8n", report["yolov8n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-batch", type=int, default=1)
    ap.add_argument("--text-batch", type=int, default=1)
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()
    report_path = OUT / "export_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if not args.skip_clip:
        export_clip(args.image_batch, args.text_batch, report)
    if not args.skip_yolo:
        export_yolo(report)
    report_path.write_text(json.dumps(report, indent=2))
    print("wrote", report_path)


if __name__ == "__main__":
    main()
