import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from footagefind.audit import log_query, read_log
from footagefind.embed import l2norm, preprocess, square_crop
from footagefind.evalmetrics import first_correct_rank, gt_coverage, summarize_ranks
from footagefind.search import Hit
from footagefind.verify import NoopVerifier, OpenAICompatVerifier, parse_yes_no, rerank


def test_tokenizer_matches_open_clip():
    oc = pytest.importorskip("open_clip")
    from footagefind.tokenizer import get_tokenizer
    texts = ["a person carrying a large bag near the gate", "Woman in a RED jacket!!", "  two   people ",
             "colourful children's toys", "ऑटो रिक्शा near the shop"]
    ours = get_tokenizer()(texts)
    ref = oc.get_tokenizer("ViT-B-32-quickgelu")(texts).numpy()
    np.testing.assert_array_equal(ours, ref)
    assert ours.dtype == np.int32


def test_preprocess_shape_and_crop_is_square():
    img = np.random.default_rng(0).integers(0, 255, (576, 768, 3), dtype=np.uint8)
    x = preprocess(img)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    c = square_crop(img, (700, 10, 760, 200))        # tall box at the right edge
    assert c.shape[0] == c.shape[1]
    assert np.allclose(np.linalg.norm(l2norm(np.ones((2, 5))), axis=1), 1.0)


def test_audit_log_appends(tmp_path):
    p = tmp_path / "a" / "audit.jsonl"
    log_query(p, "red jacket", ["vtest.avi"], 5)
    log_query(p, "white van", ["vtest.avi", "x.avi"], 3, verify=True)
    recs = read_log(p)
    assert [r["query"] for r in recs] == ["red jacket", "white van"]
    assert recs[1]["verify"] is True and "time" in recs[0]
    assert read_log(p, last=1)[0]["query"] == "white van"


def test_eval_metrics():
    gt = [{"video": "a.avi", "start": 10, "end": 12}]
    hits = [Hit("x", 50, 0.9, 50, 50, 1, "frame"), Hit("x", 12.2, 0.8, 12, 13, 2, "crop")]
    assert first_correct_rank(hits, gt, 0.25, {"x": "a.avi"}) == 2
    assert first_correct_rank(hits, gt, 0.0, {"x": "a.avi"}) is None
    s = summarize_ranks([1, 3, None, 12])
    assert s["recall@1"] == 0.25 and s["recall@5"] == 0.5 and s["recall@10"] == 0.5 and s["not_found"] == 1
    assert s["median_rank"] == 7.5
    assert gt_coverage([("a.avi", 11), ("a.avi", 20), ("b.avi", 11)], gt, 0) == pytest.approx(1 / 3)


def test_parse_yes_no():
    assert parse_yes_no("Yes.") == "yes" and parse_yes_no(" **No** ") == "no"
    assert parse_yes_no("Maybe") is None and parse_yes_no("") is None


class _StubVLM(BaseHTTPRequestHandler):
    """Mimics an OpenAI-compatible server (e.g. `geniex serve`); says yes iff prompt mentions 'red'."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        yes = "red" in content[1]["text"]
        resp = {"choices": [{"message": {"content": "Yes" if yes else "No"},
                             "logprobs": {"content": [{"token": "Yes", "logprob": -0.1 if yes else -3.0,
                                                       "top_logprobs": [{"token": "Yes", "logprob": -0.1 if yes else -3.0},
                                                                        {"token": "No", "logprob": -3.0 if yes else -0.1}]}]}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _StubVLM)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_compat_verifier_against_stub(stub_server):
    v = OpenAICompatVerifier(base_url=stub_server, timeout_s=5)
    img = np.zeros((64, 64, 3), np.uint8)
    r = v.verify(img, "a woman in a red jacket")
    assert r.answer == "yes" and r.p_yes > 0.9 and r.error is None
    r = v.verify(img, "a white van")
    assert r.answer == "no" and r.p_yes < 0.1


def test_verifier_unreachable_is_not_fatal():
    v = OpenAICompatVerifier(base_url="http://127.0.0.1:9/v1", timeout_s=1)
    r = v.verify(np.zeros((8, 8, 3), np.uint8), "x")
    assert r.p_yes is None and r.error


def test_rerank_promotes_vlm_yes(stub_server):
    hits = [Hit("v", 1, 0.30, 1, 1, 1, "frame"), Hit("v", 9, 0.29, 9, 9, 2, "frame")]

    class Picky:
        name = "picky"
        def verify(self, img, q):
            from footagefind.verify import VerifyResult
            return VerifyResult(1.0 if img[0, 0, 0] else 0.0, "yes" if img[0, 0, 0] else "no")

    imgs = [np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)]
    out = rerank(hits, imgs, "q", Picky(), weight=0.7)
    assert [h.t for h in out] == [9, 1]
    same = rerank([Hit("v", 1, 0.3, 1, 1, 1, "frame"), Hit("v", 2, 0.2, 2, 2, 2, "frame")],
                  imgs, "q", NoopVerifier(), weight=0.5)
    assert [h.t for h in same] == [1, 2]
