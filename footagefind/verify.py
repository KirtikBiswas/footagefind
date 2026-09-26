"""Optional second-stage verification with a local vision-language model.

CLIP retrieval is fast but coarse. For the top-k hits we can ask a local VLM a
yes/no question - "Does this image show: <query>?" - and re-rank.

Target on Snapdragon: Qwen3-VL-4B-Instruct served by Qualcomm GenieX
(``geniex serve``), which exposes an OpenAI-compatible HTTP endpoint on
localhost. Any OpenAI-compatible local server (llama.cpp server, vLLM,
LM Studio, Ollama's /v1) speaks the same protocol.

STATUS: the HTTP client is exercised against a stub server in the tests only.
It has NOT been run against a real VLM (none was downloaded in the build
environment). Treat the real-model path as untested.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

PROMPT = "Does this image show: {query}? Answer with a single word: yes or no."
log = logging.getLogger("footagefind.verify")
_server_proc: subprocess.Popen | None = None


@dataclass
class VerifyResult:
    p_yes: float | None      # probability-like score in [0, 1]; None = no opinion
    answer: str | None
    latency_ms: float = 0.0
    error: str | None = None


class Verifier(Protocol):
    name: str

    def verify(self, image_bgr: np.ndarray, query: str) -> VerifyResult: ...


class NoopVerifier:
    """Default: no VLM. Leaves CLIP ranking untouched."""
    name = "none"

    def verify(self, image_bgr: np.ndarray, query: str) -> VerifyResult:
        return VerifyResult(p_yes=None, answer=None)


def _encode_jpeg(image_bgr: np.ndarray, max_side: int = 768) -> str:
    h, w = image_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def parse_yes_no(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower().lstrip("*\"' ").split()
    if not t:
        return None
    w = t[0].strip(".,!:;\"'*")
    return w if w in ("yes", "no") else None


def p_yes_from_logprobs(choice: dict) -> float | None:
    """Use first-token top_logprobs when the server returns them."""
    try:
        top = choice["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    py = pn = 0.0
    for cand in top:
        tok = cand.get("token", "").strip().lower()
        if tok == "yes":
            py += math.exp(cand["logprob"])
        elif tok == "no":
            pn += math.exp(cand["logprob"])
    return py / (py + pn) if (py + pn) > 0 else None


class OpenAICompatVerifier:
    """Client for an OpenAI-compatible /v1/chat/completions endpoint on localhost."""
    name = "openai-compatible"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen3-VL-4B-Instruct",
                 timeout_s: float = 60, api_key: str = "local", **_ignored):
        # **_ignored swallows config keys meant for other layers (enabled, weight, ...)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    def verify(self, image_bgr: np.ndarray, query: str) -> VerifyResult:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3,
            "logprobs": True,
            "top_logprobs": 5,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _encode_jpeg(image_bgr)}},
                    {"type": "text", "text": PROMPT.format(query=query)},
                ],
            }],
        }
        t0 = time.perf_counter()
        try:
            resp = self._post(payload)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            return VerifyResult(None, None, (time.perf_counter() - t0) * 1000, error=str(e))
        latency = (time.perf_counter() - t0) * 1000
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
        answer = parse_yes_no(text)
        p = p_yes_from_logprobs(choice)
        if p is None and answer is not None:
            p = 1.0 if answer == "yes" else 0.0
        return VerifyResult(p, answer, latency)


def server_alive(base_url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def ensure_server(base_url: str, start_command: str = "", wait_s: float = 180.0) -> bool:
    """Lazy VLM loading: start the local VLM server only when verification is first requested.

    A 4B-parameter VLM is the largest thing FootageFind could load. On a 16 GB
    laptop it should not sit in memory while the user is only indexing or doing
    CLIP search, so it runs as a separate process that is launched on demand
    (``verify.start_command``, e.g. the ``geniex serve ...`` invocation for your
    install) and can be closed independently. Returns True if the server answers.
    """
    global _server_proc
    if server_alive(base_url):
        return True
    if not start_command:
        return False
    if _server_proc is None or _server_proc.poll() is not None:
        log.info("starting VLM server: %s", start_command)
        # Windows CreateProcess parses a command string itself; POSIX needs argv.
        _server_proc = subprocess.Popen(start_command if os.name == "nt" else shlex.split(start_command))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_alive(base_url):
            return True
        if _server_proc.poll() is not None:
            log.warning("VLM server exited with code %s", _server_proc.returncode)
            return False
        time.sleep(1.0)
    return False


def stop_server() -> None:
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
    _server_proc = None


def make_verifier(cfg: dict | None, enabled: bool | None = None) -> Verifier:
    cfg = dict(cfg or {})
    on = cfg.get("enabled", False) if enabled is None else enabled
    if not on:
        return NoopVerifier()
    start_command = cfg.pop("start_command", "")
    ensure_server(cfg.get("base_url", "http://127.0.0.1:8000/v1"), start_command, float(cfg.pop("start_wait_s", 180)))
    return OpenAICompatVerifier(**cfg)


def rerank(hits: list, images: list[np.ndarray], query: str, verifier: Verifier, weight: float = 0.5) -> list:
    """Blend min-max-normalised CLIP score with the VLM's p(yes) and re-sort.

    Hits where the VLM gave no opinion (error / unparsable) get p_yes = 0.5 so
    they neither gain nor lose relative to the CLIP order.
    """
    if not hits:
        return hits
    s = np.array([h.score for h in hits], dtype=np.float64)
    span = s.max() - s.min()
    s_norm = (s - s.min()) / span if span > 0 else np.ones_like(s)
    for h, img, sn in zip(hits, images, s_norm):
        r = verifier.verify(img, query)
        h.vlm_p_yes, h.vlm_answer = r.p_yes, r.answer
        if r.error:
            h.extras["vlm_error"] = r.error
        p = 0.5 if r.p_yes is None else r.p_yes
        h.final_score = float((1 - weight) * sn + weight * p)
    return sorted(hits, key=lambda h: -h.final_score)
