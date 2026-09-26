"""Compile + profile FootageFind's three ONNX models on Snapdragon X devices via Qualcomm AI Hub.

STATUS: NOT RUN. Written against the qai_hub 0.55.0 Python client (API
signatures checked locally with `inspect`; app.aihub.qualcomm.com was not
reachable from the build environment). Requires a free AI Hub account:

    pip install qai-hub
    qai-hub configure --api_token <YOUR_TOKEN>       # from https://app.aihub.qualcomm.com/ (Settings)

    # plan only, no network (safe to run anywhere):
    python scripts/aihub_compile_profile.py --dry-run
    # FP32 compile+profile, then w8a16 quantize -> compile -> profile, both devices:
    python scripts/aihub_compile_profile.py --precision fp32 w8a16
    # add an on-device inference job to compare outputs with local ONNX Runtime CPU:
    python scripts/aihub_compile_profile.py --precision w8a16 --check-numerics

Precision paths
  fp32   : ONNX (FP32) -> compile (HTP runs it in FP16) -> profile
  w8a16  : ONNX (FP32) -> AI Hub quantize job (INT8 weights, INT16 activations,
           ~200 calibration images from the indexed video) -> compile -> profile
  w8a8   : same with INT8 activations (smaller/faster, less accurate; see
           results/RESULTS.md for the CPU accuracy preview of each)

Outputs: results/aihub_<model>_<precision>_<device>.json (job ids/URLs, profile
summary, per-layer compute-unit counts) and results/aihub_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICES = ["Snapdragon X Elite CRD", "Snapdragon X Plus 8-Core CRD"]
MODELS = {
    # name: (onnx file, input name, shape, dtype)
    "clip_image": ("clip_image.onnx", "image", (1, 3, 224, 224), "float32"),
    "clip_text": ("clip_text.onnx", "tokens", (1, 77), "int32"),
    "yolov8n": ("yolov8n.onnx", "images", (1, 3, 640, 640), "float32"),
}
ONNX = ROOT / "models" / "onnx"
RESULTS = ROOT / "results"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def calibration(video: Path, n: int) -> dict[str, list[np.ndarray]]:
    """Per-model calibration samples (lists of single-batch arrays) from the indexed video."""
    from footagefind.calibration import (calibration_frames, clip_image_calibration, clip_text_calibration,
                                         yolo_calibration)
    from footagefind.config import load_config, resolve
    from footagefind.detect import Detector
    from footagefind.runtime import OrtModel

    cfg = load_config(overrides={"runtime": {"providers": ["CPUExecutionProvider"]}})
    det = Detector(OrtModel(resolve(cfg["models"]["yolo"]), cfg["runtime"]), **cfg["detect"])
    return {
        "clip_image": list(clip_image_calibration(calibration_frames(video, n=n, detector=det))),
        "clip_text": list(clip_text_calibration()),
        "yolov8n": list(yolo_calibration(calibration_frames(video, n=min(n, 100)))),
    }


def summarize_profile(profile: dict) -> dict:
    s = profile.get("execution_summary", {})
    layers = profile.get("execution_detail", [])
    units: dict[str, int] = {}
    for layer in layers:
        units[layer.get("compute_unit", "?")] = units.get(layer.get("compute_unit", "?"), 0) + 1
    t = s.get("estimated_inference_time")
    return {
        # AI Hub reports times in microseconds and memory in bytes.
        "estimated_inference_time_ms": None if t is None else round(t / 1000.0, 3),
        "inference_memory_peak_range_mb": [round(x / 2**20, 1) for x in s["inference_memory_peak_range"]]
        if isinstance(s.get("inference_memory_peak_range"), (list, tuple)) else None,
        "first_load_time_ms": None if s.get("first_load_time") is None else round(s["first_load_time"] / 1000.0, 1),
        "warm_load_time_ms": None if not isinstance(s.get("warm_load_time"), (int, float)) else round(s["warm_load_time"] / 1000.0, 1),
        "layers_by_compute_unit": units,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--devices", nargs="+", default=DEVICES)
    ap.add_argument("--precision", nargs="+", default=["fp32", "w8a16"], choices=["fp32", "w8a16", "w8a8"])
    ap.add_argument("--target-runtime", default="precompiled_qnn_onnx",
                    help="precompiled_qnn_onnx (ONNX wrapping a QNN context binary; loads in ORT QNN EP), "
                         "qnn_context_binary, onnx, or qnn_dlc")
    ap.add_argument("--video", default=str(ROOT / "data" / "videos" / "vtest.avi"), help="calibration source")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--check-numerics", action="store_true", help="also run an inference job and compare to ORT CPU")
    ap.add_argument("--dry-run", action="store_true", help="print the job plan; do not contact AI Hub")
    args = ap.parse_args()

    plan = [(m, p, d) for m in args.models for p in args.precision for d in args.devices]
    print(f"{len(plan)} compile+profile jobs planned (target runtime {args.target_runtime}):")
    for m, p, d in plan:
        f, name, shape, dtype = MODELS[m]
        print(f"  {m:10s} {p:6s} {d:32s} input {name}{list(shape)} {dtype}  <- models/onnx/{f}")
    calib = None
    if any(p != "fp32" for p in args.precision):
        calib = calibration(Path(args.video), args.calib_n)
        for k, v in calib.items():
            print(f"  calibration {k}: {len(v)} samples of {v[0].shape} {v[0].dtype}")
    if args.dry_run:
        print("dry run: nothing submitted.")
        return

    import qai_hub as hub
    try:
        hub.get_devices()
    except Exception as e:  # pragma: no cover - needs network + token
        sys.exit(f"AI Hub not reachable or not configured ({e}). Run `qai-hub configure --api_token ...`.")

    RESULTS.mkdir(exist_ok=True)
    summary = []
    for m in args.models:
        fname, in_name, shape, dtype = MODELS[m]
        src = ONNX / fname
        input_specs = {in_name: (shape, dtype)}
        for prec in args.precision:
            model_for_compile = str(src)
            quant_job = None
            if prec != "fp32":
                act = hub.QuantizeDtype.INT16 if prec == "w8a16" else hub.QuantizeDtype.INT8
                quant_job = hub.submit_quantize_job(
                    model=str(src), calibration_data={in_name: calib[m]},
                    weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=act,
                    name=f"footagefind-{m}-{prec}-quantize",
                )
                model_for_compile = quant_job.get_target_model()   # blocks until the QDQ ONNX is ready
                if model_for_compile is None:
                    print(f"quantize job failed for {m} {prec}: {quant_job.url}")
                    continue
            for dev_name in args.devices:
                dev = hub.Device(dev_name)
                t0 = time.time()
                cjob = hub.submit_compile_job(
                    model=model_for_compile, device=dev, input_specs=input_specs,
                    options=f"--target_runtime {args.target_runtime}",
                    name=f"footagefind-{m}-{prec}-compile",
                )
                target = cjob.get_target_model()
                rec = {"model": m, "precision": prec, "device": dev_name, "target_runtime": args.target_runtime,
                       "quantize_job": getattr(quant_job, "url", None), "compile_job": cjob.url}
                if target is None:
                    rec["status"] = "compile failed"
                else:
                    pjob = hub.submit_profile_job(model=target, device=dev, name=f"footagefind-{m}-{prec}-profile")
                    profile = pjob.download_profile()
                    rec.update({"profile_job": pjob.url, "summary": summarize_profile(profile), "profile": profile})
                    target.download(str(ONNX / f"{m}.{prec}.{slug(dev_name)}.aihub"))
                    if args.check_numerics:
                        import onnxruntime as ort
                        sample = calib[m][0] if calib else np.random.rand(*shape).astype(dtype)
                        ijob = hub.submit_inference_job(model=target, device=dev, inputs={in_name: [sample]})
                        dev_out = list(ijob.download_output_data().values())[0][0].reshape(-1)
                        ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(
                            None, {in_name: sample})[0].reshape(-1)
                        cos = float(dev_out @ ref / (np.linalg.norm(dev_out) * np.linalg.norm(ref) + 1e-12))
                        rec.update({"inference_job": ijob.url, "cosine_vs_ort_cpu_fp32": round(cos, 5)})
                    rec["status"] = "ok"
                rec["wall_s"] = round(time.time() - t0, 1)
                out = RESULTS / f"aihub_{m}_{prec}_{slug(dev_name)}.json"
                out.write_text(json.dumps(rec, indent=2, default=str))
                print("wrote", out, rec.get("summary"))
                summary.append({k: v for k, v in rec.items() if k != "profile"})
    (RESULTS / "aihub_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
