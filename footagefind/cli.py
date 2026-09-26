"""Command line interface.

    python -m footagefind index data/videos/vtest.avi
    python -m footagefind search "person carrying a white bag" -k 5
    python -m footagefind search "woman in a red jacket" --verify
    python -m footagefind runtime            # which EP each model loaded on
    python -m footagefind audit --last 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config, resolve


def _fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="footagefind", description="Offline natural-language CCTV search")
    ap.add_argument("--config", help="extra TOML merged over configs/default.toml")
    ap.add_argument("--db", help="override index database path")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index one or more videos")
    p.add_argument("videos", nargs="+")
    p.add_argument("--fps", type=float, help="sampling rate (default from config: 2)")
    p.add_argument("--no-crops", action="store_true", help="skip YOLO; embed full frames only")
    p.add_argument("--no-motion", action="store_true", help="disable motion gating")
    p.add_argument("--motion-method", choices=["diff", "mog2"])

    p = sub.add_parser("search", help="search indexed videos")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int)
    p.add_argument("--verify", action="store_true", help="re-rank top-k with local VLM (untested)")
    p.add_argument("--frames-only", action="store_true", help="ignore crop embeddings")
    p.add_argument("--json", action="store_true")

    sub.add_parser("runtime", help="show execution providers and node placement")
    p = sub.add_parser("audit", help="print the query audit log")
    p.add_argument("--last", type=int, default=20)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    overrides: dict = {}
    if getattr(args, "fps", None):
        overrides.setdefault("index", {})["sample_fps"] = args.fps
    if getattr(args, "motion_method", None):
        overrides.setdefault("motion", {})["method"] = args.motion_method
    cfg = load_config(args.config, overrides)

    if args.cmd == "audit":
        from .audit import read_log
        for rec in read_log(resolve(cfg["audit"]["log_path"]), args.last):
            print(json.dumps(rec, ensure_ascii=False))
        return 0

    from .pipeline import FootageFind
    ff = FootageFind(cfg, db_path=args.db)

    if args.cmd == "index":
        for v in args.videos:
            def prog(f: float, msg: str) -> None:
                print(f"\r[{f*100:5.1f}%] {msg}".ljust(90), end="", file=sys.stderr, flush=True)
            stats = ff.index_video(v, progress=prog, use_crops=not args.no_crops,
                                   motion_enabled=False if args.no_motion else None)
            print(file=sys.stderr)
            print(json.dumps(stats, indent=2))
    elif args.cmd == "search":
        hits = ff.search(args.query, top_k=args.top_k, verify=args.verify, use_crops=not args.frames_only)
        names = {v["id"]: v["name"] for v in ff.index.videos()}
        if args.json:
            print(json.dumps([h.as_dict() for h in hits], indent=2))
        else:
            for r, h in enumerate(hits, 1):
                vlm = f"  vlm={h.vlm_answer}({h.vlm_p_yes})" if args.verify else ""
                print(f"{r:2d}. {names.get(h.video_id, h.video_id)}  {_fmt_t(h.t)}  "
                      f"[{_fmt_t(h.t_start)}-{_fmt_t(h.t_end)}]  score={h.score:.3f}  via={h.kind}"
                      f"{'/' + h.label if h.label else ''}{vlm}")
    elif args.cmd == "runtime":
        import onnxruntime as ort
        print("onnxruntime", ort.__version__, "available providers:", ort.get_available_providers())
        for d in ff.runtime_info(load_all=True):
            print(json.dumps(d))
        for d in ff.node_placement():
            print(json.dumps(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
