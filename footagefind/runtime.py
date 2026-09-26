"""ONNX Runtime session factory: the single place where execution providers are chosen.

Every model in FootageFind (CLIP image, CLIP text, YOLO) is run through
:class:`OrtModel`, so exactly the same code path runs on an x86 CPU and on a
Snapdragon X Hexagon NPU (QNNExecutionProvider). Nothing here imports PyTorch.

Provider selection
------------------
The config lists providers in priority order, e.g.
``["QNNExecutionProvider", "CPUExecutionProvider"]``. Providers not compiled
into the installed onnxruntime build are skipped and logged. With
``disable_cpu_ep_fallback = true`` ORT refuses to create a session that would
silently place some nodes on the CPU, which is the simplest proof that a model
runs *entirely* on the NPU.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger("footagefind.runtime")

QNN = "QNNExecutionProvider"
CPU = "CPUExecutionProvider"


@dataclass
class ProviderPlan:
    requested: list[str]
    available: list[str]
    providers: list[tuple[str, dict]]           # what we pass to InferenceSession
    skipped: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [p for p, _ in self.providers]


def qnn_provider_options(rt_cfg: dict) -> dict:
    opts = {
        "backend_path": rt_cfg.get("qnn_backend_path", "QnnHtp.dll"),
        "htp_performance_mode": rt_cfg.get("qnn_htp_performance_mode", "burst"),
        "htp_graph_finalization_optimization_mode": "3",
        # FP32 graphs are executed in FP16 on the HTP; QDQ graphs run INT8/INT16.
        "enable_htp_fp16_precision": "1",
    }
    return opts


def select_providers(rt_cfg: dict, available: list[str] | None = None) -> ProviderPlan:
    """Pure function: decide which providers to request given what is installed."""
    requested = list(rt_cfg.get("providers") or [CPU])
    env = os.environ.get("FOOTAGEFIND_PROVIDERS")
    if env:  # e.g. FOOTAGEFIND_PROVIDERS=CPUExecutionProvider
        requested = [p.strip() for p in env.split(",") if p.strip()]
    available = list(ort.get_available_providers() if available is None else available)
    plan = ProviderPlan(requested=requested, available=available, providers=[])
    for name in requested:
        if name not in available:
            plan.skipped.append(name)
            continue
        opts = qnn_provider_options(rt_cfg) if name == QNN else {}
        plan.providers.append((name, opts))
    if not plan.providers:
        if rt_cfg.get("disable_cpu_ep_fallback"):
            raise RuntimeError(
                f"None of the requested providers {requested} are available "
                f"(installed onnxruntime offers {available}) and CPU fallback is disabled."
            )
        plan.providers.append((CPU, {}))
    return plan


def _session_options(rt_cfg: dict, plan: ProviderPlan, ctx_path: Path | None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    threads = int(rt_cfg.get("intra_op_threads", 0) or 0)
    if threads:
        so.intra_op_num_threads = threads
    if rt_cfg.get("disable_cpu_ep_fallback"):
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if ctx_path is not None and QNN in plan.names:
        # Ask the QNN EP to dump a compiled context binary on first load so later
        # launches skip graph finalisation (seconds -> milliseconds).
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", str(ctx_path))
        so.add_session_config_entry("ep.context_embed_mode", "0")
    return so


def context_cache_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_ctx.onnx")


class OrtModel:
    """Thin wrapper around an ``ort.InferenceSession`` that records where it runs."""

    def __init__(self, model_path: str | os.PathLike, rt_cfg: dict, name: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found. Run `python scripts/export_onnx.py` first."
            )
        self.name = name or self.model_path.stem
        self.rt_cfg = rt_cfg
        self.plan = select_providers(rt_cfg)
        load_path, ctx_path = self.model_path, None
        if rt_cfg.get("qnn_context_cache") and QNN in self.plan.names:
            cached = context_cache_path(self.model_path)
            if cached.exists():
                load_path = cached          # pre-compiled QNN context binary
            else:
                ctx_path = cached           # generate it during this load
        so = _session_options(rt_cfg, self.plan, ctx_path)
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(load_path), sess_options=so, providers=self.plan.providers)
        self.load_seconds = time.perf_counter() - t0
        self.loaded_from = load_path
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.latencies_ms: list[float] = []
        log.info(
            "%s: loaded %s on %s in %.2fs (requested %s, skipped %s, cpu_fallback_disabled=%s)",
            self.name, load_path.name, self.session.get_providers(), self.load_seconds,
            self.plan.requested, self.plan.skipped, bool(rt_cfg.get("disable_cpu_ep_fallback")),
        )

    @property
    def providers(self) -> list[str]:
        """Providers registered on the live session, highest priority first."""
        return self.session.get_providers()

    @property
    def primary_provider(self) -> str:
        return self.providers[0]

    @property
    def batch_size(self) -> int:
        dim = self.inputs[0].shape[0]
        return dim if isinstance(dim, int) and dim > 0 else 1

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        t0 = time.perf_counter()
        out = self.session.run(None, feed)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    def describe(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else None
        return {
            "model": self.name,
            "file": self.loaded_from.name,
            "session_providers": self.providers,
            "primary_provider": self.primary_provider,
            "requested": self.plan.requested,
            "skipped_unavailable": self.plan.skipped,
            "cpu_fallback_disabled": bool(self.rt_cfg.get("disable_cpu_ep_fallback")),
            "load_seconds": round(self.load_seconds, 3),
            "runs": 0 if lat is None else int(lat.size),
            "mean_ms": None if lat is None else round(float(lat.mean()), 2),
            "p95_ms": None if lat is None else round(float(np.percentile(lat, 95)), 2),
        }


def dummy_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "int" in inp.type:
            feed[inp.name] = np.zeros(shape, dtype=np.int64 if "int64" in inp.type else np.int32)
        else:
            feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed


def probe_node_placement(model_path: str | os.PathLike, rt_cfg: dict) -> dict:
    """Run once with ORT profiling and count which EP executed each graph node.

    ``session.get_providers()`` only tells you which providers were *registered*;
    the profile tells you where kernels actually ran. On a Snapdragon device a
    fully offloaded model shows (almost) all nodes under QNNExecutionProvider.
    """
    plan = select_providers(rt_cfg)
    with tempfile.TemporaryDirectory() as tmp:
        so = _session_options(rt_cfg, plan, None)
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(tmp) / "probe")
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=plan.providers)
        sess.run(None, dummy_feed(sess))
        prof = sess.end_profiling()
        with open(prof) as f:
            events = json.load(f)
    counts: Counter = Counter()
    for ev in events:
        if ev.get("cat") == "Node" and ev.get("name", "").endswith("_kernel_time"):
            counts[ev.get("args", {}).get("provider", "unknown")] += 1
    return {"model": Path(model_path).name, "session_providers": sess.get_providers(), "nodes_by_provider": dict(counts)}
