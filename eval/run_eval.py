"""FootageFind evaluation on this machine's CPU (ONNX Runtime CPUExecutionProvider).

    python eval/run_eval.py                 # full run (~20-30 min on a 4-core VM)
    python eval/run_eval.py --quick         # skip quantized variants and the idle-padded clip

Measures, for the queries in eval/queries.json:
  * Recall@1/5/10, MRR and median rank (merged/de-duplicated result list)
    - with vs without YOLO crops, with vs without motion gating
    - FP32 vs INT8 (w8a8) vs w8a16 CLIP encoders
  * indexing wall time and sampled frames/s for every configuration
  * motion gating on a synthetic idle-padded copy of vtest (see below)
  * per-model mean and p95 latency via ONNX Runtime on this CPU
  * YOLO w8a16 vs FP32 detection agreement

Writes results/cpu_eval.json and results/RESULTS.md. Every number is labelled
with the hardware it was measured on. Nothing here estimates NPU performance.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from footagefind.config import load_config  # noqa: E402
from footagefind.detect import Detector, letterbox  # noqa: E402
from footagefind.embed import ClipEmbedder, preprocess  # noqa: E402
from footagefind.evalmetrics import first_correct_rank, gt_coverage, summarize_ranks  # noqa: E402
from footagefind.hwinfo import hardware_info, hardware_label  # noqa: E402
from footagefind.ingest import sample_frames  # noqa: E402
from footagefind.pipeline import FootageFind  # noqa: E402
from footagefind.runtime import OrtModel  # noqa: E402
from footagefind.search import search_vector  # noqa: E402

VIDEOS = ROOT / "data" / "videos"
WORK = ROOT / "data" / "eval"
ONNX = ROOT / "models" / "onnx"
RESULTS = ROOT / "results"
QUERIES = json.loads((ROOT / "eval" / "queries.json").read_text())
TOL = float(QUERIES["tolerance_s"])
IDLE_PAD_S = 60.0


CONFIG_PATH: str | None = None   # set by --config (e.g. configs/snapdragon.toml on device)


def cpu_cfg(models: dict | None = None) -> dict:
    """Default: force the CPU EP. With --config, use that file's providers unchanged."""
    over: dict = {} if CONFIG_PATH else {"runtime": {"providers": ["CPUExecutionProvider"]}}
    if models:
        over["models"] = models
    return load_config(CONFIG_PATH, overrides=over)


# ------------------------------------------------------------------------ helpers
def build_index(name: str, cfg: dict, videos: list[Path], use_crops: bool, motion: bool,
                shared_models: dict | None = None) -> tuple[FootageFind, list[dict]]:
    db = WORK / f"{name}.db"
    for p in (db, WORK / f"{name}_thumbs"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    cfg = copy.deepcopy(cfg)
    cfg["index"]["thumbs_dir"] = str(WORK / f"{name}_thumbs")
    cfg["audit"]["log_path"] = str(WORK / "audit_eval.jsonl")
    ff = FootageFind(cfg, db_path=db)
    if shared_models is not None:
        ff._models = shared_models
    stats = []
    for v in videos:
        s = ff.index_video(v, use_crops=use_crops, motion_enabled=motion, make_proxy=False)
        stats.append(s)
        print(f"  [{name}] {v.name}: {s['index_wall_s']}s, kept {s['motion']['frames_kept']}/"
              f"{s['motion']['frames_seen']}, crops {s['crops_embedded']}", flush=True)
    return ff, stats


def evaluate(ff: FootageFind, use_crops: bool = True, gt_shift: dict | None = None) -> dict:
    """Rank every query against the index. gt_shift: {video_name: (new_name, offset_s)}."""
    name_of = {v["id"]: v["name"] for v in ff.index.videos()}
    meta = ff.index.meta_arrays()
    per_query, ranks = [], {"all": [], "event": [], "scene": []}
    stamps = sorted({(name_of[v], round(t, 3)) for v, t in zip(meta["video_id"], meta["t"])})
    for q in QUERIES["queries"]:
        gt = q["gt"]
        if gt_shift:
            gt = [dict(g, video=gt_shift[g["video"]][0], start=g["start"] + gt_shift[g["video"]][1],
                       end=g["end"] + gt_shift[g["video"]][1]) if g["video"] in gt_shift else g for g in gt]
        t0 = time.perf_counter()
        qvec = ff.encode_query(q["query"])
        hits = search_vector(ff.index, qvec, top_k=None, merge_seconds=float(ff.cfg["search"]["merge_seconds"]),
                             use_crops=use_crops, meta=meta)
        ms = (time.perf_counter() - t0) * 1000
        r = first_correct_rank(hits, gt, TOL, name_of)
        ranks["all"].append(r)
        ranks[q["category"]].append(r)
        per_query.append({
            "id": q["id"], "query": q["query"], "category": q["category"], "rank": r,
            "top1": {"video": name_of[hits[0].video_id], "t": hits[0].t, "score": round(hits[0].score, 4),
                     "via": hits[0].kind} if hits else None,
            "n_results": len(hits), "gt_coverage": round(gt_coverage(stamps, gt, TOL), 4),
            "query_ms": round(ms, 1),
        })
    out = {k: summarize_ranks(v) for k, v in ranks.items() if v}
    out["random_baseline_recall@1_approx"] = round(float(np.mean([p["gt_coverage"] for p in per_query])), 4)
    out["per_query"] = per_query
    return out


def make_idle_padded(src: Path, dst: Path, pad_s: float = IDLE_PAD_S, noise_sigma: float = 2.0) -> Path:
    """Synthetic 'quiet night' clip: frozen first/last frame + mild sensor noise around vtest."""
    if dst.exists():
        return dst
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    rng = np.random.default_rng(0)

    def idle(img):
        for _ in range(int(pad_s * fps)):
            noise = rng.normal(0, noise_sigma, img.shape)
            wr.write(np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    idle(frames[0])
    for f in frames:
        wr.write(f)
    idle(frames[-1])
    wr.release()
    return dst


def bench_model(path: Path, feed_fn, n: int = 50, warmup: int = 5) -> dict:
    cfg = cpu_cfg()
    m = OrtModel(path, cfg["runtime"], name=path.stem)
    feed = feed_fn(m)
    for _ in range(warmup):
        m.run(feed)
    m.latencies_ms.clear()
    for _ in range(n):
        m.run(feed)
    lat = np.array(m.latencies_ms)
    return {"model": path.name, "provider": m.primary_provider, "runs": n,
            "mean_ms": round(float(lat.mean()), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "size_mb": round(path.stat().st_size / 1e6, 1)}


def detection_agreement(fp32: Path, other: Path, video: Path, n_frames: int = 40) -> dict:
    cfg = cpu_cfg()
    da = Detector(OrtModel(fp32, cfg["runtime"]), **cfg["detect"])
    db = Detector(OrtModel(other, cfg["runtime"]), **cfg["detect"])
    frames = [s.image for s in sample_frames(video, 0.5)][:n_frames]
    tp = na = nb = 0
    for f in frames:
        a, b = da(f), db(f)
        na += len(a); nb += len(b)
        used = set()
        for x in a:
            best, bi = 0.0, None
            for j, y in enumerate(b):
                if j in used or y.cls != x.cls:
                    continue
                ix1, iy1 = max(x.box[0], y.box[0]), max(x.box[1], y.box[1])
                ix2, iy2 = min(x.box[2], y.box[2]), min(x.box[3], y.box[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ua = (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]) + (y.box[2] - y.box[0]) * (y.box[3] - y.box[1]) - inter
                iou = inter / ua if ua else 0
                if iou > best:
                    best, bi = iou, j
            if bi is not None and best >= 0.5:
                used.add(bi); tp += 1
    return {"frames": len(frames), "fp32_detections": na, "other_detections": nb, "matched_iou>=0.5": tp,
            "recall_vs_fp32": round(tp / na, 4) if na else None, "precision_vs_fp32": round(tp / nb, 4) if nb else None}


# ------------------------------------------------------------------------- report
def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.3f}"
    return str(v)


def write_markdown(res: dict, path: Path) -> None:
    hw = res["hardware_label"]
    L = [
        "# FootageFind - measured results",
        "",
        f"All numbers below were measured by `python eval/run_eval.py` on **{hw}**"
        + (" (virtual machine)" if res["hardware"].get("virtualized") else "") + f", on {res['date']}.",
        "No number in this file was measured on, or extrapolated to, Snapdragon hardware.",
        "",
        f"Test data: `vtest.avi` (79.5 s, 768x576, OpenCV sample) + `indoor_desk.avi` (6.6 s, 1920x1080, opencv_extra) "
        f"indexed together; {res['n_queries']} queries with manually labelled ground truth (`eval/queries.json`), "
        f"hit tolerance +/-{TOL}s, sampling {res['sample_fps']} fps, merge window {res['merge_seconds']} s. "
        "Ranks are computed over the full merged result list.",
        "",
        "## Retrieval accuracy (all queries)",
        "",
        "| Configuration | CLIP precision | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank | Event R@1 | Event R@5 | Scene R@1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in res["retrieval"]:
        a, e, s = row["metrics"]["all"], row["metrics"].get("event", {}), row["metrics"].get("scene", {})
        L.append(f"| {row['label']} | {row['precision']} | {fmt(a['recall@1'])} | {fmt(a['recall@5'])} | "
                 f"{fmt(a['recall@10'])} | {fmt(a['mrr'])} | {fmt(a['median_rank'])} | {fmt(e.get('recall@1'))} | "
                 f"{fmt(e.get('recall@5'))} | {fmt(s.get('recall@1'))} |")
    L += ["", f"Approximate random-ranking Recall@1 (mean share of indexed timestamps inside ground truth): "
          f"{res['retrieval'][0]['metrics']['random_baseline_recall@1_approx']:.3f}. Several queries have broad "
          "ground truth (e.g. q03, q08, and the scene-level queries), so the per-query table matters more than the average.", ""]

    L += ["## Indexing cost", "",
          "| Configuration | Videos | Wall time (s) | Sampled frames | Kept by motion gate | Crops embedded | Sampled frames/s | x real time |",
          "|---|---|---|---|---|---|---|---|"]
    for row in res["retrieval"] + res.get("idle_padded", {}).get("rows", []):
        st = row["index_stats"]
        wall = sum(s["index_wall_s"] for s in st)
        seen = sum(s["motion"]["frames_seen"] for s in st)
        kept = sum(s["motion"]["frames_kept"] for s in st)
        crops = sum(s["crops_embedded"] for s in st)
        dur = sum(s["duration_s"] for s in st)
        L.append(f"| {row['label']} | {', '.join(s['video'] for s in st)} | {wall:.1f} | {seen} | {kept} | {crops} | "
                 f"{seen / wall:.2f} | {dur / wall:.2f} |")
    L += ["", "x real time = seconds of video indexed per wall-clock second. Model load time is excluded; "
          "YOLO is always FP32 so the CLIP precision rows isolate the CLIP change.", ""]

    if res.get("idle_padded"):
        ip = res["idle_padded"]
        L += ["## Motion gating on a synthetic idle-padded clip", "",
              f"`vtest.avi` never has a static moment (people walk through every frame), so the gate keeps 100% of its frames. "
              f"To show what the gate does on typical CCTV (long idle stretches) we built **a synthetic clip**: "
              f"{ip['pad_s']:.0f} s of the frozen first frame + vtest + {ip['pad_s']:.0f} s of the frozen last frame, "
              f"with Gaussian noise (sigma={ip['noise_sigma']}) added to the frozen parts. Ground truth is shifted by "
              f"+{ip['pad_s']:.0f} s. Real DVR footage has compression artefacts, lighting changes and swaying trees, "
              "so real skip ratios will be lower than on this idealised clip.", "",
              "| Configuration | Recall@1 | Recall@5 | Recall@10 | Median rank | Frames skipped | Index wall time (s) |",
              "|---|---|---|---|---|---|---|"]
        for row in ip["rows"]:
            a = row["metrics"]["all"]
            st = row["index_stats"]
            skipped = sum(s["motion"]["frames_skipped"] for s in st)
            seen = sum(s["motion"]["frames_seen"] for s in st)
            L.append(f"| {row['label']} | {fmt(a['recall@1'])} | {fmt(a['recall@5'])} | {fmt(a['recall@10'])} | "
                     f"{fmt(a['median_rank'])} | {skipped}/{seen} | {sum(s['index_wall_s'] for s in st):.1f} |")
        L.append("")

    L += [f"## Per-model latency (ONNX Runtime, {res['latency'][0]['provider'] if res['latency'] else 'n/a'})", "",
          f"Batch 1, static shapes, {res['latency'][0]['runs']} timed runs after 5 warm-up runs, on {hw}.", "",
          "| Model file | Precision | Size (MB) | Mean (ms) | p95 (ms) |", "|---|---|---|---|---|"]
    for b in res["latency"]:
        prec = "w8a16 QDQ" if "w8a16" in b["model"] else "INT8 w8a8 QDQ" if "int8" in b["model"] else "FP32"
        L.append(f"| {b['model']} | {prec} | {b['size_mb']} | {b['mean_ms']} | {b['p95_ms']} |")
    L += ["", "w8a16 models are the NPU target. x86 CPUs have no native 16-bit-activation kernels, so ORT emulates the "
          "QDQ ops and they run *slower* than FP32 here; their CPU latency says nothing about NPU latency.", ""]
    q = res.get("query_latency")
    if q:
        L += [f"End-to-end query latency (text encode + cosine over {q['index_items']} vectors + merge), FP32, "
              f"{q['n']} queries: mean {q['mean_ms']} ms, p95 {q['p95_ms']} ms.", ""]
    if res.get("yolo_w8a16_agreement"):
        y = res["yolo_w8a16_agreement"]
        L += ["## YOLOv8n w8a16 vs FP32 (CPU, detection agreement)", "",
              f"On {y['frames']} vtest frames: FP32 {y['fp32_detections']} detections, w8a16 {y['other_detections']}, "
              f"{y['matched_iou>=0.5']} matched at IoU>=0.5 with the same class "
              f"(recall vs FP32 {y['recall_vs_fp32']}, precision vs FP32 {y['precision_vs_fp32']}).", ""]
    if res.get("embedding_fidelity"):
        L += ["## Quantized CLIP embedding fidelity (mean cosine to FP32 embeddings)", "",
              "| Encoder | INT8 w8a8 | w8a16 |", "|---|---|---|"]
        for k, v in res["embedding_fidelity"].items():
            L.append(f"| {k} | {v.get('int8')} | {v.get('w8a16')} |")
        L.append("")

    base = next(r for r in res["retrieval"] if r["key"] == "fp32_crops_motion")
    fo = next((r for r in res["retrieval"] if r["key"] == "fp32_frames_motion"), None)
    L += ["## Per-query ranks", "", "| id | query | GT coverage | rank (crops) | rank (frames only) | top-1 (crops) |",
          "|---|---|---|---|---|---|"]
    fo_rank = {p["id"]: p["rank"] for p in fo["metrics"]["per_query"]} if fo else {}
    for p in base["metrics"]["per_query"]:
        t1 = p["top1"]
        L.append(f"| {p['id']} | {p['query']} | {p['gt_coverage']:.2f} | {fmt(p['rank'])} | {fmt(fo_rank.get(p['id']))} | "
                 f"{t1['video']} @ {t1['t']:.1f}s ({t1['via']}) |")
    L += ["", "## Snapdragon X (AI Hub / on-device) - to be measured", "",
          "Nothing below has been measured yet. Fill in from `results/aihub_*.json` (scripts/aihub_compile_profile.py) "
          "and on-device runs of `python eval/run_eval.py --config configs/snapdragon.toml`.", "",
          "| Model | Precision | Device | Runtime | Compute unit | Mean latency (ms) | Peak memory (MB) | Source |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ["CLIP ViT-B/32 image", "CLIP ViT-B/32 text", "YOLOv8n"]:
        for dev in ["Snapdragon X Elite CRD", "Snapdragon X Plus 8-Core CRD"]:
            L.append(f"| {m} | w8a16 | {dev} | | | | | |")
    L += ["", "| On-device end-to-end | Value |", "|---|---|",
          "| Indexing throughput (sampled frames/s) | |", "| Query latency (ms) | |",
          "| Recall@1 / @5 / @10 (same queries) | |", "| Share of nodes on QNNExecutionProvider | |", ""]
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip quantized variants and idle-padded clip")
    ap.add_argument("--out", default=str(RESULTS / "cpu_eval.json"))
    ap.add_argument("--config", help="extra TOML (e.g. configs/snapdragon.toml); its providers are used as-is")
    ap.add_argument("--md", default=str(RESULTS / "RESULTS.md"))
    args = ap.parse_args()
    global CONFIG_PATH
    CONFIG_PATH = args.config
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    videos = [VIDEOS / name for name in QUERIES["videos"]]
    for v in videos:
        if not v.exists():
            sys.exit(f"missing {v}; run scripts/download_assets.py first")

    hw = hardware_info()
    eps = ",".join(cpu_cfg()["runtime"]["providers"])
    res = {"date": time.strftime("%Y-%m-%d"), "hardware": hw, "hardware_label": hardware_label(hw, eps),
           "config": args.config,
           "n_queries": len(QUERIES["queries"]), "tolerance_s": TOL}
    base_cfg = cpu_cfg()
    res["sample_fps"] = base_cfg["index"]["sample_fps"]
    res["merge_seconds"] = base_cfg["search"]["merge_seconds"]
    print("Hardware:", res["hardware_label"], flush=True)

    shared: dict = {}
    res["retrieval"] = []
    grid = [("fp32_crops_motion", "Frames + YOLO crops, motion gate on (default)", True, True),
            ("fp32_crops_nomotion", "Frames + YOLO crops, motion gate off", True, False),
            ("fp32_frames_motion", "Full frames only, motion gate on", False, True),
            ("fp32_frames_nomotion", "Full frames only, motion gate off", False, False)]
    ff_default = None
    for key, label, crops, motion in grid:
        print(f"== {label}", flush=True)
        ff, stats = build_index(key, base_cfg, videos, crops, motion, shared)
        m = evaluate(ff, use_crops=crops)
        res["retrieval"].append({"key": key, "label": label, "precision": "FP32", "use_crops": crops,
                                 "motion": motion, "index_stats": stats, "metrics": m})
        print(f"   R@1={m['all']['recall@1']} R@5={m['all']['recall@5']} R@10={m['all']['recall@10']} "
              f"median={m['all']['median_rank']}", flush=True)
        if key == "fp32_crops_motion":
            ff_default = ff

    # end-to-end query latency on the default index
    lat = []
    for _ in range(3):
        for q in QUERIES["queries"]:
            t0 = time.perf_counter()
            ff_default.search(q["query"], top_k=10, audit_log=False)
            lat.append((time.perf_counter() - t0) * 1000)
    res["query_latency"] = {"n": len(lat), "mean_ms": round(float(np.mean(lat)), 1),
                            "p95_ms": round(float(np.percentile(lat, 95)), 1), "index_items": ff_default.index.count()}

    if not args.quick:
        fidelity: dict = {"clip_image": {}, "clip_text": {}}
        test_imgs = [s.image for s in sample_frames(videos[0], 0.25)] + [s.image for s in sample_frames(videos[1], 1)]
        texts = [q["query"] for q in QUERIES["queries"]]
        ref = ClipEmbedder(OrtModel(ONNX / "clip_image.onnx", base_cfg["runtime"]),
                           OrtModel(ONNX / "clip_text.onnx", base_cfg["runtime"]))
        ri, rt = ref.embed_images(test_imgs), ref.embed_texts(texts)
        for prec in ("int8", "w8a16"):
            img_p, txt_p = ONNX / f"clip_image.{prec}.onnx", ONNX / f"clip_text.{prec}.onnx"
            if not (img_p.exists() and txt_p.exists()):
                print(f"skip {prec}: run scripts/quantize_onnx.py", flush=True)
                continue
            e = ClipEmbedder(OrtModel(img_p, base_cfg["runtime"]), OrtModel(txt_p, base_cfg["runtime"]))
            fidelity["clip_image"][prec] = round(float((e.embed_images(test_imgs) * ri).sum(1).mean()), 4)
            fidelity["clip_text"][prec] = round(float((e.embed_texts(texts) * rt).sum(1).mean()), 4)
            label = {"int8": "INT8 w8a8", "w8a16": "w8a16 (NPU target numerics)"}[prec]
            print(f"== CLIP {label}", flush=True)
            cfg = cpu_cfg({"clip_image": str(img_p), "clip_text": str(txt_p)})
            qshared = {"yolo": shared["yolo"]}
            ff, stats = build_index(f"{prec}_crops_motion", cfg, videos, True, True, qshared)
            m = evaluate(ff)
            res["retrieval"].append({"key": f"{prec}_crops_motion", "label": f"Frames + crops, motion on, CLIP {label}",
                                     "precision": label, "use_crops": True, "motion": True,
                                     "index_stats": stats, "metrics": m})
            print(f"   R@1={m['all']['recall@1']} R@5={m['all']['recall@5']} R@10={m['all']['recall@10']}", flush=True)
        res["embedding_fidelity"] = fidelity

        print("== idle-padded synthetic clip", flush=True)
        padded = make_idle_padded(videos[0], WORK / "vtest_idle_padded.avi")
        shift = {"vtest.avi": ("vtest_idle_padded.avi", IDLE_PAD_S)}
        rows = []
        for key, label, motion in [("idle_motion", "Idle-padded vtest + indoor, crops, motion gate on", True),
                                   ("idle_nomotion", "Idle-padded vtest + indoor, crops, motion gate off", False)]:
            ff, stats = build_index(key, base_cfg, [padded, videos[1]], True, motion, shared)
            m = evaluate(ff, gt_shift=shift)
            rows.append({"key": key, "label": label, "index_stats": stats, "metrics": m})
            print(f"   R@1={m['all']['recall@1']} skipped={stats[0]['motion']['frames_skipped']}", flush=True)
        res["idle_padded"] = {"pad_s": IDLE_PAD_S, "noise_sigma": 2.0, "rows": rows}

    print("== latency", flush=True)
    frame = next(iter(sample_frames(videos[0], 1))).image
    img_feed = lambda m: {m.inputs[0].name: preprocess(frame)[None].astype(np.float32)}  # noqa: E731
    tok = ClipEmbedder(None, None).tokenizer(["a woman in a red jacket"])
    txt_feed = lambda m: {m.inputs[0].name: tok.astype(np.int64 if "int64" in m.inputs[0].type else np.int32)}  # noqa: E731
    lb, _, _ = letterbox(frame, 640)
    yolo_feed = lambda m: {m.inputs[0].name: (lb[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)}  # noqa: E731
    res["latency"] = []
    for fname, feed in [("clip_image.onnx", img_feed), ("clip_text.onnx", txt_feed), ("yolov8n.onnx", yolo_feed),
                        ("clip_image.int8.onnx", img_feed), ("clip_text.int8.onnx", txt_feed),
                        ("clip_image.w8a16.onnx", img_feed), ("clip_text.w8a16.onnx", txt_feed),
                        ("yolov8n.w8a16.onnx", yolo_feed)]:
        if args.quick and ("int8" in fname or "w8a16" in fname):
            continue
        if (ONNX / fname).exists():
            b = bench_model(ONNX / fname, feed)
            res["latency"].append(b)
            print("  ", b, flush=True)
    if not args.quick and (ONNX / "yolov8n.w8a16.onnx").exists():
        res["yolo_w8a16_agreement"] = detection_agreement(ONNX / "yolov8n.onnx", ONNX / "yolov8n.w8a16.onnx", videos[0])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    write_markdown(res, Path(args.md))
    print("wrote", args.out, "and", args.md, flush=True)


if __name__ == "__main__":
    main()
