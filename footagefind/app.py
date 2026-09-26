"""Gradio UI.   python -m footagefind.app  [--config configs/snapdragon.toml] [--port 7860]

Runs fully offline on 127.0.0.1. Video playback uses a browser-friendly WebM
copy created at index time (DVR exports are often AVI/H.264 variants that
browsers will not play).
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

import cv2

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # offline product: no usage pings
import gradio as gr  # noqa: E402
import onnxruntime as ort  # noqa: E402

from . import __version__  # noqa: E402
from .audit import read_log  # noqa: E402
from .config import REPO_ROOT, load_config, resolve  # noqa: E402
from .hwinfo import hardware_info, hardware_label  # noqa: E402
from .pipeline import FootageFind  # noqa: E402

log = logging.getLogger("footagefind.app")
VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov", ".dav", ".h264", ".264", ".webm", ".mpg", ".mpeg", ".ts"}

CSS = """
#ff-header h1 {margin-bottom: 0}
#ff-header p {margin-top: 4px; opacity: .8}
.ff-badge {display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; margin-right:6px;
           background: var(--color-accent-soft); border: 1px solid var(--border-color-accent)}
#ff-player video {width: 100%; max-height: 460px; background: #000; border-radius: 8px}
#ff-player .ff-cap {font-size: 13px; opacity: .85; margin-top: 4px}
"""


def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ff = FootageFind(cfg)
        self.lock = threading.Lock()
        self.videos_dir = REPO_ROOT / "data" / "videos"
        self.annot_dir = resolve(cfg["index"]["thumbs_dir"]).parent / "annotated"
        self.hw = hardware_info()

    # ------------------------------------------------------------------ helpers
    def local_videos(self) -> list[str]:
        if not self.videos_dir.exists():
            return []
        return sorted(str(p.relative_to(REPO_ROOT)) for p in self.videos_dir.rglob("*")
                      if p.suffix.lower() in VIDEO_EXT)

    def indexed_choices(self) -> list[tuple[str, str]]:
        return [(f"{v['name']}  ({fmt_t(v['duration'] or 0)})", v["id"]) for v in self.ff.index.videos()]

    def library_md(self) -> str:
        vids = self.ff.index.videos()
        if not vids:
            return "_No videos indexed yet._"
        rows = ["| Video | Length | Frames kept / sampled | Crops | Index time | Indexed |", "|---|---|---|---|---|---|"]
        for v in vids:
            s = v["stats"] or {}
            m = s.get("motion", {})
            rows.append(f"| {v['name']} | {fmt_t(v['duration'] or 0)} | {m.get('frames_kept', '?')} / "
                        f"{m.get('frames_seen', '?')} | {s.get('crops_embedded', '?')} | {s.get('index_wall_s', '?')} s | "
                        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(v['indexed_at']))} |")
        return "\n".join(rows)

    def runtime_md(self) -> str:
        info = self.ff.runtime_info()
        lines = [f"**Hardware:** {hardware_label(self.hw)}", "",
                 f"**onnxruntime {ort.__version__}** - execution providers compiled into this build: "
                 f"`{', '.join(ort.get_available_providers())}`", "",
                 f"**Requested order:** `{' > '.join(self.cfg['runtime']['providers'])}`  |  "
                 f"CPU fallback disabled: `{bool(self.cfg['runtime'].get('disable_cpu_ep_fallback'))}`", ""]
        if not info:
            lines.append("_Models load lazily; index a video or run a search to load them._")
            return "\n".join(lines)
        lines += ["| Model | File | Running on | Registered providers | Load (s) | Runs | Mean (ms) | p95 (ms) |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in info:
            on = d["primary_provider"]
            badge = "NPU (QNN HTP)" if on == "QNNExecutionProvider" else "CPU" if on == "CPUExecutionProvider" else on
            lines.append(f"| {d['model']} | `{d['file']}` | **{badge}** | {', '.join(d['session_providers'])} | "
                         f"{d['load_seconds']} | {d['runs']} | {d['mean_ms'] or '-'} | {d['p95_ms'] or '-'} |")
        return "\n".join(lines)

    def provider_badges(self) -> str:
        info = {d["model"]: d["primary_provider"] for d in self.ff.runtime_info()}
        if not info:
            return ""
        short = {"QNNExecutionProvider": "NPU", "CPUExecutionProvider": "CPU"}
        return " ".join(f"<span class='ff-badge'>{k}: {short.get(v, v)}</span>" for k, v in info.items())

    def annotated_thumb(self, hit) -> str:
        """Thumbnail with the matching detection box drawn (crop hits)."""
        if hit.kind != "crop" or not hit.bbox or not hit.thumb:
            return hit.thumb
        out = self.annot_dir / f"{hit.item_id}.jpg"
        if not out.exists():
            img = cv2.imread(hit.thumb)
            v = self.ff.index.video(hit.video_id)
            s = img.shape[1] / v["width"]
            x1, y1, x2, y2 = (int(round(c * s)) for c in hit.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img)
        return str(out)

    def player_html(self, hit=None, names=None) -> str:
        if hit is None:
            return "<div id='ff-player'><div class='ff-cap'>Click a result to play it from that moment.</div></div>"
        v = self.ff.index.video(hit.video_id)
        src = v.get("proxy_path") or v["path"]
        start = max(0.0, hit.t - 1.0)          # 1 s lead-in before the best-matching moment
        url = f"/gradio_api/file={quote(str(Path(src).resolve()))}#t={start:.1f}"
        cap = (f"<b>{html.escape(v['name'])}</b> - playing from {fmt_t(start)} (best match {fmt_t(hit.t)}, "
               f"range {fmt_t(hit.t_start)}-{fmt_t(hit.t_end)}, CLIP score {hit.score:.3f}"
               + (f", matched a <i>{html.escape(hit.label)}</i> crop" if hit.label else ", matched the full frame") + ")")
        return (f"<div id='ff-player'><video key='{hit.item_id}' src='{url}' controls autoplay muted playsinline "
                f"preload='auto'></video><div class='ff-cap'>{cap}</div></div>")

    # ----------------------------------------------------------------- handlers
    def do_index(self, upload, local_choice, fps, use_crops, motion, progress=gr.Progress()):
        path = None
        if upload:
            src = Path(upload if isinstance(upload, str) else upload.name)
            dst = self.videos_dir / "uploads" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            path = dst
        elif local_choice:
            path = REPO_ROOT / local_choice
        if path is None:
            raise gr.Error("Upload a video or pick one from the list.")
        self.ff.cfg["index"]["sample_fps"] = float(fps)
        with self.lock:
            progress(0, desc="loading models")
            stats = self.ff.index_video(path, progress=lambda f, m: progress(f, desc=m),
                                        use_crops=use_crops, motion_enabled=motion)
        m = stats["motion"]
        md = (f"**Indexed `{stats['video']}`** ({fmt_t(stats['duration_s'])}) in **{stats['index_wall_s']} s** "
              f"({stats['realtime_factor']}x real time).  \n"
              f"Sampled {m['frames_seen']} frames at {stats['sample_fps']} fps; motion gate kept {m['frames_kept']} and "
              f"**skipped {m['frames_skipped']}** ({m['skip_ratio']*100:.0f}%). "
              f"Embedded {stats['frames_embedded']} frames + {stats['crops_embedded']} detection crops.  \n"
              f"Stage time: decode+gate {stats['timings_s']['decode_gate_s']} s (own thread), detect "
              f"{stats['timings_s']['detect_s']} s, embed {stats['timings_s']['embed_s']} s.  \n"
              f"Models ran on: " + ", ".join(f"{k} -> `{v}`" for k, v in stats["providers"].items()))
        choices = self.indexed_choices()
        return (md, self.library_md(), gr.update(choices=choices, value=[c[1] for c in choices]),
                self.runtime_md(), self.provider_badges())

    def do_search(self, query, video_ids, top_k, verify):
        query = (query or "").strip()
        if not query:
            raise gr.Error("Type what you are looking for, e.g. 'woman in a red jacket'.")
        if self.ff.index.count() == 0:
            raise gr.Error("Index a video first.")
        t0 = time.perf_counter()
        hits = self.ff.search(query, top_k=int(top_k), video_ids=video_ids or None, verify=verify)
        ms = (time.perf_counter() - t0) * 1000
        names = {v["id"]: v["name"] for v in self.ff.index.videos()}
        gallery = []
        for r, h in enumerate(hits, 1):
            cap = f"{fmt_t(h.t)} | {Path(names.get(h.video_id, '?')).stem} | {h.score:.2f}"
            if verify:
                cap += f"  VLM: {h.vlm_answer or 'n/a'}"
            gallery.append((self.annotated_thumb(h), cap))
        status = (f"{len(hits)} moments for **\"{html.escape(query)}\"** in {ms:.0f} ms "
                  f"(logged to `{self.cfg['audit']['log_path']}`)")
        if verify and hits and all(h.vlm_answer is None for h in hits):
            err = hits[0].extras.get("vlm_error", "no answer")
            status += f"  \n**VLM verification unavailable** ({html.escape(str(err))[:120]}); CLIP order kept."
        first = self.player_html(hits[0], names) if hits else self.player_html()
        return gallery, hits, status, first, self.runtime_md(), self.provider_badges()

    def do_select(self, hits, evt: gr.SelectData):
        if not hits or evt.index is None:
            return self.player_html()
        return self.player_html(hits[evt.index])

    def do_audit(self):
        recs = read_log(resolve(self.cfg["audit"]["log_path"]), last=100)
        return [[r["time"], r.get("os_user", ""), r["query"], ", ".join(r["videos"]), r["n_results"],
                 "yes" if r.get("verify") else "no"] for r in reversed(recs)]

    def do_probe(self):
        rows = self.ff.node_placement()
        return "\n".join(["| Model | Nodes by execution provider (from ORT profiling) |", "|---|---|"] +
                         [f"| {r['model']} | {r['nodes_by_provider']} |" for r in rows])

    # ---------------------------------------------------------------------- UI
    def build(self) -> gr.Blocks:
        with gr.Blocks(title="FootageFind", analytics_enabled=False) as demo:
            gr.HTML(
                "<div id='ff-header'><h1>FootageFind</h1>"
                "<p>Search your own CCTV exports in plain language - fully offline. "
                "No face recognition, no identity matching; every search is logged locally.</p></div>"
            )
            badges = gr.HTML(self.provider_badges())
            hits_state = gr.State([])
            with gr.Tabs():
                with gr.Tab("Search"):
                    with gr.Row():
                        query = gr.Textbox(label="What are you looking for?", scale=5,
                                           placeholder="e.g. person carrying a large bag near the gate")
                        btn = gr.Button("Search", variant="primary", scale=1)
                    with gr.Row():
                        vids = gr.Dropdown(choices=self.indexed_choices(), value=[c[1] for c in self.indexed_choices()],
                                           multiselect=True, label="Videos", scale=4)
                        topk = gr.Slider(4, 24, value=int(self.cfg["search"]["top_k"]), step=1, label="Results", scale=2)
                        verify = gr.Checkbox(value=False, label="Verify top results with local VLM (experimental, untested)",
                                             scale=2)
                    status = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=3):
                            gallery = gr.Gallery(label="Matching moments (click to play)", columns=4, height=520,
                                                 object_fit="contain", allow_preview=False)
                        with gr.Column(scale=2):
                            player = gr.HTML(self.player_html())
                with gr.Tab("Index videos"):
                    with gr.Row():
                        with gr.Column():
                            upload = gr.File(label="Upload a DVR/NVR export", file_types=sorted(VIDEO_EXT))
                            local = gr.Dropdown(choices=self.local_videos(), label="...or pick a file from data/videos")
                        with gr.Column():
                            fps = gr.Slider(0.5, 5, value=float(self.cfg["index"]["sample_fps"]), step=0.5,
                                            label="Frames sampled per second")
                            crops = gr.Checkbox(value=bool(self.cfg["index"]["use_crops"]),
                                                label="Detect people/vehicles/bags and embed crops (better recall, slower)")
                            motion = gr.Checkbox(value=bool(self.cfg["motion"]["enabled"]),
                                                 label="Motion gate: skip frames where nothing moved")
                            idx_btn = gr.Button("Index video", variant="primary")
                    idx_status = gr.Markdown()
                    library = gr.Markdown(self.library_md())
                with gr.Tab("Runtime"):
                    runtime = gr.Markdown(self.runtime_md())
                    with gr.Row():
                        load_btn = gr.Button("Load all models now")
                        probe_btn = gr.Button("Probe node placement (ORT profiling)")
                    probe_out = gr.Markdown()
                with gr.Tab("Audit log"):
                    gr.Markdown("Every search is appended to a local JSON-lines file. Nothing leaves this machine.")
                    audit_btn = gr.Button("Refresh")
                    audit_tbl = gr.Dataframe(headers=["time", "OS user", "query", "videos", "results", "VLM verify"],
                                             value=self.do_audit(), wrap=True)
                with gr.Tab("About"):
                    gr.Markdown(ABOUT.format(version=__version__))

            outs = [gallery, hits_state, status, player, runtime, badges]
            btn.click(self.do_search, [query, vids, topk, verify], outs)
            query.submit(self.do_search, [query, vids, topk, verify], outs)
            gallery.select(self.do_select, [hits_state], [player])
            idx_btn.click(self.do_index, [upload, local, fps, crops, motion], [idx_status, library, vids, runtime, badges])
            load_btn.click(lambda: (self.ff.runtime_info(load_all=True), self.runtime_md())[1], None, runtime)
            probe_btn.click(self.do_probe, None, probe_out)
            audit_btn.click(self.do_audit, None, audit_tbl)
        return demo


ABOUT = """
### FootageFind {version}
Offline natural-language search over exported CCTV footage for shops, housing societies and small offices.

**Pipeline:** OpenCV decode (own thread) -> motion gate -> YOLOv8n person/vehicle/bag detection ->
CLIP ViT-B/32 embeddings of full frames and crops -> SQLite + numpy index -> CLIP text query -> cosine
similarity -> temporal de-duplication -> optional local VLM yes/no verification.
All models run through ONNX Runtime; on a Snapdragon X laptop the QNN execution provider places them on the
Hexagon NPU, on other machines they run on the CPU.

**What it does not do:** no face recognition, no identity matching, no cloud calls, no telemetry.
It finds *moments that look like a description*; results are leads to review, not evidence of who someone is.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--lazy", action="store_true", help="load models on first use instead of at startup")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    app = App(cfg)
    if not args.lazy:
        for d in app.ff.runtime_info(load_all=True):
            log.info("model %s -> %s (registered %s)", d["model"], d["primary_provider"], d["session_providers"])
    demo = app.build()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, allowed_paths=[str(REPO_ROOT / "data")],
        theme=gr.themes.Soft(), css=CSS,
    )


if __name__ == "__main__":
    main()
