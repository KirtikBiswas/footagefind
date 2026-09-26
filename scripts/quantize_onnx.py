"""Static QDQ quantization with ONNX Runtime (runs on any CPU, no device needed).

    python scripts/quantize_onnx.py --video data/videos/vtest.avi            # both modes
    python scripts/quantize_onnx.py --modes int8                             # CPU INT8 only

Two outputs per model, both calibrated on ~200 images from the given video
(frames + YOLO person crops for CLIP-image, letterboxed frames for YOLO) and on
generic captions for CLIP-text (never the eval queries):

* ``*.int8.onnx``  - w8a8 (INT8 weights, UINT8 activations), Entropy
  calibration, only MatMul/Gemm/Conv quantized. CPU-oriented; this is what the
  FP32-vs-INT8 comparison in eval/run_eval.py uses.
* ``*.w8a16.onnx`` - the ONNX Runtime QNN recipe (``qnn_preprocess_model`` +
  ``get_qnn_qdq_config``): INT8 weights, UINT16 activations, *every* op
  quantized so the whole graph can be placed on the Hexagon HTP. These are the
  files configs/snapdragon.toml points at. On an x86 CPU they run through
  QDQ emulation and are slow; their CPU latency is meaningless, but their
  retrieval accuracy on CPU is a faithful preview of the NPU numerics.

In a small sweep on the CLIP image encoder (64 calibration images,
MatMul/Gemm/Conv only; mean cosine to FP32 embeddings on held-out frames)
w8a8 MinMax gave 0.82, w8a8 Percentile(99.99) 0.76, w8a8 Entropy 0.85 and
w8a16 MinMax 0.998, which is why w8a16 is the NPU target.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize, quantize_static)
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from footagefind.calibration import (calibration_frames, clip_image_calibration,  # noqa: E402
                                     clip_text_calibration, yolo_calibration)
from footagefind.config import load_config, resolve  # noqa: E402
from footagefind.detect import Detector  # noqa: E402
from footagefind.runtime import OrtModel  # noqa: E402

ONNX = ROOT / "models" / "onnx"


class Reader(CalibrationDataReader):
    def __init__(self, name: str, data: np.ndarray):
        self.name, self.data, self.i = name, data, 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return {self.name: self.data[self.i - 1]}

    def rewind(self):
        self.i = 0


def resize_scales_to_sizes(model: "onnx.ModelProto") -> int:
    """Rewrite Resize(scales=const float) as Resize(sizes=const int64).

    The QNN QDQ config quantizes *every* float tensor, including Resize's
    `scales` input. [1, 1, 2, 2] in UINT16 dequantizes to [0.99998, 0.99998, 2, 2];
    ORT then computes floor(256 * 0.99998) = 255 channels and silently drops a
    feature map. That wrecked the first w8a16 YOLOv8n (12% of FP32 detections).
    Integer `sizes` are never quantized. Needs static shapes (which we have).
    """
    from onnx import numpy_helper, shape_inference

    inferred = shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    consts = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    inits = {i.name: i for i in model.graph.initializer}
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Resize" or len(node.input) < 3 or not node.input[2]:
            continue
        src = node.input[2]
        if src in consts:
            scales = numpy_helper.to_array(consts[src].attribute[0].t)
        elif src in inits:
            scales = numpy_helper.to_array(inits[src])
        else:
            continue
        in_shape = shapes.get(node.input[0])
        if not in_shape or 0 in in_shape:
            continue
        sizes = np.floor(np.array(in_shape) * scales).astype(np.int64)
        name = node.name + "_sizes"
        model.graph.initializer.append(numpy_helper.from_array(sizes, name))
        del node.input[2:]
        node.input.extend(["", name])       # Resize(X, roi="", scales="", sizes)
        node.input[2] = ""
        changed += 1
    return changed


def quantize_int8(src: Path, dst: Path, input_name: str, data: np.ndarray) -> dict:
    t0 = time.time()
    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    ops = ["MatMul", "Gemm", "Conv"]
    quantize_static(
        str(pre), str(dst), Reader(input_name, data), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
        calibrate_method=CalibrationMethod.Entropy, op_types_to_quantize=ops,
        extra_options={"WeightSymmetric": True},
    )
    pre.unlink(missing_ok=True)
    return {"mode": "int8 (w8a8)", "src": src.name, "dst": dst.name, "calibration_samples": int(len(data)),
            "calibration": "Entropy", "op_types": ops, "seconds": round(time.time() - t0, 1),
            "size_mb": round(dst.stat().st_size / 1e6, 1)}


def quantize_qnn_w8a16(src: Path, dst: Path, input_name: str, data: np.ndarray) -> dict:
    t0 = time.time()
    src_name = src.name
    pre = dst.with_suffix(".pre.onnx")
    m = onnx.load(str(src))
    n_resize = resize_scales_to_sizes(m)
    if len(m.metadata_props) or n_resize:
        # Ultralytics writes metadata_props that make qnn_preprocess_model crash
        # (TypeError in save_and_reload_optimize_model, ORT 1.30); strip them first.
        del m.metadata_props[:]
        src = dst.with_suffix(".nometa.onnx")
        onnx.save(m, str(src))
    changed = qnn_preprocess_model(str(src), str(pre), fuse_layernorm=True)
    model_in = pre if changed else src
    qcfg = get_qnn_qdq_config(
        str(model_in), Reader(input_name, data), calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8, per_channel=True,
    )
    quantize(str(model_in), str(dst), qcfg)
    pre.unlink(missing_ok=True)
    dst.with_suffix(".nometa.onnx").unlink(missing_ok=True)
    return {"mode": "qnn w8a16", "src": src_name, "dst": dst.name, "calibration_samples": int(len(data)),
            "calibration": "MinMax", "op_types": "all (QNN QDQ config)",
            "resize_scales_rewritten_to_sizes": n_resize, "seconds": round(time.time() - t0, 1),
            "size_mb": round(dst.stat().st_size / 1e6, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(ROOT / "data" / "videos" / "vtest.avi"))
    ap.add_argument("--n", type=int, default=200, help="calibration images")
    ap.add_argument("--modes", default="int8,w8a16")
    ap.add_argument("--models", default="clip_image,clip_text,yolov8n")
    args = ap.parse_args()
    modes, models = args.modes.split(","), args.models.split(",")

    cfg = load_config(overrides={"runtime": {"providers": ["CPUExecutionProvider"]}})
    det = Detector(OrtModel(resolve(cfg["models"]["yolo"]), cfg["runtime"]), **cfg["detect"])
    data = {}
    if "clip_image" in models:
        data["clip_image"] = ("image", clip_image_calibration(calibration_frames(args.video, n=args.n, detector=det)))
    if "clip_text" in models:
        data["clip_text"] = ("tokens", clip_text_calibration())
    if "yolov8n" in models:
        data["yolov8n"] = ("images", yolo_calibration(calibration_frames(args.video, n=min(args.n, 100))))

    report_path = ONNX / "quantize_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["calibration_video"] = Path(args.video).name
    for name, (inp, arr) in data.items():
        src = ONNX / f"{name}.onnx"
        if "int8" in modes and name != "yolov8n":   # INT8 CPU comparison is for CLIP only
            report[f"{name}.int8"] = quantize_int8(src, ONNX / f"{name}.int8.onnx", inp, arr)
            print(json.dumps(report[f"{name}.int8"]))
        if "w8a16" in modes:
            report[f"{name}.w8a16"] = quantize_qnn_w8a16(src, ONNX / f"{name}.w8a16.onnx", inp, arr)
            print(json.dumps(report[f"{name}.w8a16"]))
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
