# HANDOFF - FootageFind prototype

Strictly factual status as of 2026-09-26, for writing the competition documents. Anything not listed under
"actually run" was **not** run. Build environment: cloud Linux VM, **Intel Xeon @ 2.80 GHz, 4 vCPU (KVM guest),
15.7 GiB RAM, no GPU, no NPU**, Python 3.11.15, ONNX Runtime 1.30.0 (CPU and Azure EPs only). Hugging Face and
app.aihub.qualcomm.com were blocked by the environment's egress policy; GitHub was reachable.

## 1. What was built

| Path | What it is |
|---|---|
| `footagefind/runtime.py` | ONNX Runtime session factory. Provider list from config (`QNNExecutionProvider` with `backend_path=QnnHtp.dll`, then `CPUExecutionProvider`); skips/logs unavailable EPs; `disable_cpu_ep_fallback` strict mode (drops the CPU EP, because ORT 1.30 rejects CPU EP + that flag); QNN context-binary caching (`ep.context_enable`); `probe_node_placement()` counts nodes per EP from an ORT profile. |
| `footagefind/ingest.py` | OpenCV decode, fixed-rate sampling (default 2 fps), stable video id, VP8/WebM browser proxy. |
| `footagefind/motion.py` | Motion gate: frame differencing vs last kept frame (default) or MOG2; 30 s heartbeat; skip statistics. |
| `footagefind/detect.py` | YOLOv8n ONNX pre/post-processing (letterbox, class filter person/vehicles/bags, NMS). |
| `footagefind/embed.py` | CLIP ViT-B/32 image + text encoders on ORT; square padded crops; static-batch padding. |
| `footagefind/tokenizer.py` + `assets/` | Vendored, torch-free CLIP BPE tokenizer (byte-identical to open_clip, tested). |
| `footagefind/index.py` | SQLite (metadata + float32 blobs) + numpy brute-force cosine; optional sqlite-vec backend (installed cleanly, v0.1.9, tested equal to numpy). |
| `footagefind/search.py` | max-over-(frame, crops) per timestamp, then greedy temporal NMS anchored on the best frame (`merge_seconds`, default 3). |
| `footagefind/verify.py` | `Verifier` protocol, `NoopVerifier`, `OpenAICompatVerifier` (chat/completions with base64 JPEG, yes/no parsing, top-logprob p(yes)), `rerank()`, lazy server start (`verify.start_command`). |
| `footagefind/audit.py` | Append-only JSONL audit log (time, OS user, query, videos, n results, verify flag). |
| `footagefind/pipeline.py` | `FootageFind` engine: decode+gate producer thread -> bounded queue -> detect/embed/index; search; stats. |
| `footagefind/cli.py`, `__main__.py` | `python -m footagefind index/search/runtime/audit`. |
| `footagefind/app.py` | Gradio 6 UI: Search (results grid with detection boxes, click to play from the timestamp), Index videos (upload or pick, progress bar), Runtime (EP per model, latencies, node placement probe), Audit log, About. Analytics disabled. |
| `footagefind/calibration.py`, `evalmetrics.py`, `hwinfo.py`, `config.py` | Calibration data, metrics, hardware labels, TOML config loading. |
| `configs/default.toml`, `configs/snapdragon.toml` | CPU/dev config; Snapdragon config (QNN, w8a16 models, strict NPU, context cache). |
| `scripts/export_onnx.py` | Static-shape ONNX export (opset 17) of CLIP image `[1,3,224,224]`, CLIP text `[1,77]` int32 (with -inf mask made finite), YOLOv8n `[1,3,640,640]`. |
| `scripts/quantize_onnx.py` | ORT static QDQ: `*.int8.onnx` (w8a8, Entropy, MatMul/Gemm/Conv) and `*.w8a16.onnx` (ORT QNN recipe, all ops; Resize scales rewritten to sizes). |
| `scripts/aihub_compile_profile.py` | qai_hub client: fp32 / w8a16 / w8a8 quantize -> compile (`precompiled_qnn_onnx`) -> profile on "Snapdragon X Elite CRD" and "Snapdragon X Plus 8-Core CRD", optional inference-job numerics check; writes `results/aihub_*.json`. **NOT RUN.** |
| `scripts/download_assets.py`, `scripts/capture_screenshots.py` | SHA-256-checked downloads; Playwright screenshot driver. |
| `eval/queries.json`, `eval/evidence/` (28 JPEGs), `eval/run_eval.py` | 16 queries with hand-labelled ground truth + 5 dropped queries with reasons; evidence frames; evaluation harness. |
| `results/RESULTS.md`, `results/cpu_eval.json` | Generated evaluation output (CPU only). |
| `tests/` (41 tests) | motion gate, index (numpy + sqlite-vec), search merge/de-dup, provider selection (incl. strict mode), tokenizer parity, VLM client vs stub server, lazy server start, audit log, metrics, Resize rewrite regression. |
| `docs/SNAPDRAGON.md`, `docs/screenshots/` | Device guide; 6 UI screenshots. |
| `README.md`, `requirements.txt`, `requirements-snapdragon.txt`, `LICENSE` (MIT) | |

## 2. What was actually run here (exact commands, summarised output)

```bash
python scripts/download_assets.py            # CLIP laion400m_e32 .pt, yolov8n.pt, vtest.avi, 1920x1080.avi (renamed indoor_desk.avi); SHA-256 recorded
python scripts/export_onnx.py                # clip_image.onnx 351.6 MB, clip_text.onnx 254.2 MB, yolov8n.onnx 12.8 MB;
                                             # ORT vs PyTorch max abs diff 3.1e-7 (image), 2.1e-7 (text)
python scripts/quantize_onnx.py              # int8: clip_image 89.3 MB, clip_text 140.7 MB; w8a16: 89.9 / 90.4 / 3.7 MB (yolov8n)
python -m footagefind index data/videos/vtest.avi data/videos/indoor_desk.avi
python -m footagefind search "woman in a red jacket" -k 5     # (and other queries, during development)
python -m pytest -q                          # 41 passed
python eval/run_eval.py                      # -> results/cpu_eval.json, results/RESULTS.md (~35 min)
python -m footagefind.app & python scripts/capture_screenshots.py --chromium /opt/pw-browsers/chromium-1194/chrome-linux/chrome
python scripts/aihub_compile_profile.py --dry-run --precision fp32 w8a16   # builds calibration (200/60/100 samples), prints 12 planned jobs, submits nothing
```

Things discovered and fixed while running (all have tests or eval evidence):
* `torch.onnx.export` failed on `aten::_native_multi_head_attention`; fixed by disabling the MHA fast path for export.
* w8a16 CLIP text encoder was useless (cosine 0.37) because of the `-inf` causal mask; the mask is now `-100` (cosine 0.9994).
* w8a16 YOLOv8n kept only 12% of FP32 detections because the QNN QDQ config quantized Resize `scales`
  (1.0 -> 0.99998 -> 255 of 256 channels). Rewriting to integer `sizes` fixed it (301/301 detections matched).
* `qnn_preprocess_model` crashes on Ultralytics' ONNX metadata; metadata is stripped first.
* ORT 1.30 refuses `session.disable_cpu_ep_fallback=1` when the CPU EP is registered; strict mode now registers only accelerator EPs.
* Earlier eval runs: run 1 lost its output because I deleted `results/` mid-run (log: `results/logs/eval_run1_failed_write.log`);
  run 2 used the broken w8a16 YOLO (`results/logs/eval_run2_before_yolo_fix.log`, `RESULTS_run2_before_yolo_fix.md`).
  Only run 3 is reported (`results/logs/eval_run3_final.log`). Quantization log: `results/logs/quantize.log`.

## 3. All measured numbers (copy of results/RESULTS.md)

Every number below was measured on the VM described above (CPU only).

### results/RESULTS.md (verbatim)

All numbers below were measured by `python eval/run_eval.py` on **Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, 15.7 GiB RAM, Linux 6.18.44-fc-v37 (x86_64), onnxruntime 1.30.0 CPUExecutionProvider** (virtual machine), on 2026-09-26.
No number in this file was measured on, or extrapolated to, Snapdragon hardware.

Test data: `vtest.avi` (79.5 s, 768x576, OpenCV sample) + `indoor_desk.avi` (6.6 s, 1920x1080, opencv_extra) indexed together; 16 queries with manually labelled ground truth (`eval/queries.json`), hit tolerance +/-0.25s, sampling 2.0 fps, merge window 3.0 s. Ranks are computed over the full merged result list.

#### Retrieval accuracy (all queries)

| Configuration | CLIP precision | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank | Event R@1 | Event R@5 | Scene R@1 |
|---|---|---|---|---|---|---|---|---|---|
| Frames + YOLO crops, motion gate on (default) | FP32 | 0.938 | 0.938 | 1.00 | 0.945 | 1.00 | 1.00 | 1.00 | 0.667 |
| Frames + YOLO crops, motion gate off | FP32 | 0.938 | 0.938 | 1.00 | 0.945 | 1.00 | 1.00 | 1.00 | 0.667 |
| Full frames only, motion gate on | FP32 | 0.562 | 0.875 | 0.938 | 0.641 | 1.00 | 0.462 | 0.846 | 1.00 |
| Full frames only, motion gate off | FP32 | 0.562 | 0.875 | 0.938 | 0.641 | 1.00 | 0.462 | 0.846 | 1.00 |
| Frames + crops, motion on, CLIP INT8 w8a8 | INT8 w8a8 | 0.750 | 0.938 | 1.00 | 0.837 | 1.00 | 0.769 | 0.923 | 0.667 |
| Frames + crops, motion on, CLIP w8a16 (NPU target numerics) | w8a16 (NPU target numerics) | 0.938 | 1.00 | 1.00 | 0.953 | 1.00 | 1.00 | 1.00 | 0.667 |
| Frames + crops, motion on, CLIP + YOLO all w8a16 (full NPU numerics) | w8a16 (CLIP + YOLO) | 0.875 | 0.938 | 1.00 | 0.913 | 1.00 | 0.923 | 1.00 | 0.667 |

Approximate random-ranking Recall@1 (mean share of indexed timestamps inside ground truth): 0.179. Several queries have broad ground truth (e.g. q03, q08, and the scene-level queries), so the per-query table matters more than the average.

#### Indexing cost

| Configuration | Videos | Wall time (s) | Sampled frames | Kept by motion gate | Crops embedded | Sampled frames/s | x real time |
|---|---|---|---|---|---|---|---|
| Frames + YOLO crops, motion gate on (default) | vtest.avi, indoor_desk.avi | 76.9 | 173 | 173 | 1195 | 2.25 | 1.12 |
| Frames + YOLO crops, motion gate off | vtest.avi, indoor_desk.avi | 76.6 | 173 | 173 | 1195 | 2.26 | 1.12 |
| Full frames only, motion gate on | vtest.avi, indoor_desk.avi | 10.3 | 173 | 173 | 0 | 16.78 | 8.35 |
| Full frames only, motion gate off | vtest.avi, indoor_desk.avi | 10.9 | 173 | 173 | 0 | 15.81 | 7.87 |
| Frames + crops, motion on, CLIP INT8 w8a8 | vtest.avi, indoor_desk.avi | 56.7 | 173 | 173 | 1195 | 3.05 | 1.52 |
| Frames + crops, motion on, CLIP w8a16 (NPU target numerics) | vtest.avi, indoor_desk.avi | 391.5 | 173 | 173 | 1195 | 0.44 | 0.22 |
| Frames + crops, motion on, CLIP + YOLO all w8a16 (full NPU numerics) | vtest.avi, indoor_desk.avi | 403.3 | 173 | 173 | 1183 | 0.43 | 0.21 |
| Idle-padded vtest + indoor, crops, motion gate on | vtest_idle_padded.avi, indoor_desk.avi | 80.0 | 413 | 177 | 1223 | 5.16 | 2.57 |
| Idle-padded vtest + indoor, crops, motion gate off | vtest_idle_padded.avi, indoor_desk.avi | 180.9 | 413 | 413 | 2804 | 2.28 | 1.14 |

x real time = seconds of video indexed per wall-clock second. Model load time is excluded; YOLO is FP32 in every row except the 'all w8a16' row, so the CLIP precision rows isolate the CLIP change.

#### Motion gating on a synthetic idle-padded clip

`vtest.avi` never has a static moment (people walk through every frame), so the gate keeps 100% of its frames. To show what the gate does on typical CCTV (long idle stretches) we built **a synthetic clip**: 60 s of the frozen first frame + vtest + 60 s of the frozen last frame, with Gaussian noise (sigma=2.0) added to the frozen parts. Ground truth is shifted by +60 s. Real DVR footage has compression artefacts, lighting changes and swaying trees, so real skip ratios will be lower than on this idealised clip.

| Configuration | Recall@1 | Recall@5 | Recall@10 | Median rank | Frames skipped | Index wall time (s) |
|---|---|---|---|---|---|---|
| Idle-padded vtest + indoor, crops, motion gate on | 0.875 | 0.938 | 1.00 | 1.00 | 236/413 | 80.0 |
| Idle-padded vtest + indoor, crops, motion gate off | 0.812 | 0.938 | 1.00 | 1.00 | 0/413 | 180.9 |

#### Per-model latency (ONNX Runtime, CPUExecutionProvider)

Batch 1, static shapes, 50 timed runs after 5 warm-up runs, on Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, 15.7 GiB RAM, Linux 6.18.44-fc-v37 (x86_64), onnxruntime 1.30.0 CPUExecutionProvider.

| Model file | Precision | Size (MB) | Mean (ms) | p95 (ms) |
|---|---|---|---|---|
| clip_image.onnx | FP32 | 351.6 | 40.67 | 49.59 |
| clip_text.onnx | FP32 | 254.2 | 36.98 | 45.13 |
| yolov8n.onnx | FP32 | 12.8 | 44.35 | 52.2 |
| clip_image.int8.onnx | INT8 w8a8 QDQ | 89.3 | 24.36 | 31.86 |
| clip_text.int8.onnx | INT8 w8a8 QDQ | 140.7 | 24.03 | 26.77 |
| clip_image.w8a16.onnx | w8a16 QDQ | 89.9 | 268.95 | 298.49 |
| clip_text.w8a16.onnx | w8a16 QDQ | 90.4 | 153.87 | 173.61 |
| yolov8n.w8a16.onnx | w8a16 QDQ | 3.7 | 125.6 | 149.15 |

w8a16 models are the NPU target. x86 CPUs have no native 16-bit-activation kernels, so ORT emulates the QDQ ops and they run *slower* than FP32 here; their CPU latency says nothing about NPU latency.

End-to-end query latency (text encode + cosine over 1368 vectors + merge), FP32, 48 queries: mean 102.9 ms, p95 161.4 ms.

#### YOLOv8n w8a16 vs FP32 (CPU, detection agreement)

On 40 vtest frames: FP32 301 detections, w8a16 301, 301 matched at IoU>=0.5 with the same class (recall vs FP32 1.0, precision vs FP32 1.0).

#### Quantized CLIP embedding fidelity (mean cosine to FP32 embeddings)

| Encoder | INT8 w8a8 | w8a16 |
|---|---|---|
| clip_image | 0.8458 | 0.9987 |
| clip_text | 0.7893 | 0.9994 |

#### Per-query ranks

| id | query | GT coverage | rank (crops) | rank (frames only) | top-1 (crops) |
|---|---|---|---|---|---|
| q01 | a man and a woman walking together on the grass | 0.09 | 1 | 5 | vtest.avi @ 55.0s (crop) |
| q02 | a woman with curly blonde hair in a long black coat walking alone on the grass | 0.20 | 1 | 1 | vtest.avi @ 68.5s (crop) |
| q03 | a woman in a red jacket | 0.40 | 1 | 1 | vtest.avi @ 78.5s (crop) |
| q04 | a person in a white hooded jacket | 0.12 | 1 | 10 | vtest.avi @ 12.5s (crop) |
| q05 | a man in a blue jacket and grey trousers | 0.10 | 1 | 5 | vtest.avi @ 66.5s (crop) |
| q06 | a group of three people standing and talking next to a signpost | 0.08 | 1 | 1 | vtest.avi @ 4.0s (crop) |
| q07 | a person right next to the camera tripod on the grass | 0.05 | 1 | 4 | vtest.avi @ 60.0s (crop) |
| q08 | people walking on the grass | 0.28 | 1 | 4 | vtest.avi @ 55.0s (crop) |
| q09 | a man in a red jacket walking on the grass | 0.13 | 1 | 1 | vtest.avi @ 59.0s (crop) |
| q10 | a man in a red jacket walking side by side with a blonde woman on the road | 0.27 | 1 | 4 | vtest.avi @ 9.5s (crop) |
| q11 | a man holding a colourful ball | 0.03 | 1 | 1 | indoor_desk.avi @ 4.5s (crop) |
| q12 | a man with his hand on top of a toy clown | 0.02 | 1 | - | indoor_desk.avi @ 2.0s (crop) |
| q13 | a man reaching across the table to pick up a ball | 0.02 | 1 | 1 | indoor_desk.avi @ 4.5s (crop) |
| q14 | colourful children's toys on a wooden table | 0.08 | 1 | 1 | indoor_desk.avi @ 0.0s (frame) |
| q15 | a white van parked in front of a brick building | 0.92 | 1 | 1 | vtest.avi @ 73.5s (crop) |
| q16 | a man in a striped short-sleeved shirt | 0.08 | 8 | 1 | vtest.avi @ 15.0s (crop) |

#### Snapdragon X (AI Hub / on-device) - to be measured

Nothing below has been measured yet. Fill in from `results/aihub_*.json` (scripts/aihub_compile_profile.py) and on-device runs of `python eval/run_eval.py --config configs/snapdragon.toml`.

| Model | Precision | Device | Runtime | Compute unit | Mean latency (ms) | Peak memory (MB) | Source |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 image | w8a16 | Snapdragon X Elite CRD | | | | | |
| CLIP ViT-B/32 image | w8a16 | Snapdragon X Plus 8-Core CRD | | | | | |
| CLIP ViT-B/32 text | w8a16 | Snapdragon X Elite CRD | | | | | |
| CLIP ViT-B/32 text | w8a16 | Snapdragon X Plus 8-Core CRD | | | | | |
| YOLOv8n | w8a16 | Snapdragon X Elite CRD | | | | | |
| YOLOv8n | w8a16 | Snapdragon X Plus 8-Core CRD | | | | | |

| On-device end-to-end | Value |
|---|---|
| Indexing throughput (sampled frames/s) | |
| Query latency (ms) | |
| Recall@1 / @5 / @10 (same queries) | |
| Share of nodes on QNNExecutionProvider | |

Additional numbers visible in the screenshots (same VM, from the app, not the eval harness): indexing vtest.avi
through the UI took 82.5 s (159 frames kept of 159, 1181 crops) and indoor_desk.avi 3.8 s. The Runtime tab's
per-model mean/p95 are running averages that include first-call warm-up, so they are higher than the benchmark.

## 4. What was NOT run, and why

| Item | Status | Why |
|---|---|---|
| Anything on Snapdragon hardware / Windows on ARM | **Not run** | No device available. `configs/snapdragon.toml`, `requirements-snapdragon.txt` and `docs/SNAPDRAGON.md` are untested. |
| QNNExecutionProvider / QnnHtp.dll | **Not run** | Only x86 onnxruntime (CPU EP) exists here. Provider selection is unit-tested with simulated provider lists; the QNN provider options and context-cache session keys have never been passed to a real QNN EP. |
| Qualcomm AI Hub compile/profile/quantize (`scripts/aihub_compile_profile.py`) | **Not run** (only `--dry-run`) | Needs an AI Hub API token; app.aihub.qualcomm.com was blocked. Written against the installed qai_hub 0.55.0 (signatures checked with `inspect`), not against live docs. No `results/aihub_*.json` exist. |
| VLM verification (Qwen3-VL-4B-Instruct via GenieX `geniex serve`) | **Not run** | No VLM downloaded (per instructions; Hugging Face blocked). Only the HTTP client was tested, against a stub OpenAI-compatible server in `tests/test_misc.py`. The GenieX command line was not verified; `verify.start_command` is left empty. The `--verify` checkbox in the UI shows "unavailable" when no server answers. |
| NPU latency, NPU memory, NPU power | **Not measured** | No hardware. The table "Snapdragon X (AI Hub / on-device) - to be measured" is intentionally empty. Do not fill it with estimates. |
| Real Indian CCTV footage (night IR, DVR mosaics, .dav files) | **Not tested** | Only two public OpenCV test clips were available. |

## 5. Known bugs and limitations

* Evaluation is tiny: 16 queries, 86 s of video, 2 clips, labelled by one annotator (me) from contact sheets. Several
  queries have broad ground truth (q03 red jacket covers ~40% of indexed timestamps; q15 van is every vtest frame).
  Treat results as a sanity check, not a benchmark.
* The INT8/w8a16 calibration images come from vtest.avi, which is also the eval video (no labels used, but it is
  still in-domain calibration).
* Motion gate: 0 frames skipped on vtest (constant movement). The skip benefit is shown only on a synthetic
  idle-padded clip with Gaussian noise; real sensor/compression noise, lighting changes and trees are not tested.
* Merge window (3 s) can hide a weaker moment close to a stronger one (e.g. q12 is not found at any rank in the
  frames-only index for this reason).
* A scene-level query failed at rank 1 with crops (q16 "striped short-sleeved shirt" ranked 8: a vtest person crop scored higher).
* CLIP ViT-B/32 is weak at counting and spatial relations; results are leads, not evidence.
* Timestamps are file-relative; no DVR clock OCR. Proprietary DVR formats depend on the local OpenCV/FFmpeg build.
* The text encoder contains ArgMax + Gather (EOT pooling); HTP support for these in strict mode is unverified.
* On x86, w8a16 QDQ models are ~7x slower than FP32 (emulation); that is expected and not a regression.
* `opencv-python-headless`/Gradio wheel availability for win_arm64 is unverified.
* YOLOv8 is AGPL-3.0 (flagged in README); a commercial build needs a licence or another detector.
* The audit log is append-only in the app but is a plain file; it is not tamper-proof.
* Playback in the UI relies on a VP8/WebM proxy generated at index time (browser-verified in headless Chromium).

## 6. Screenshots (captured with Playwright from the running app, CPU EP)

* `docs/screenshots/01_indexing_progress.png` - indexing vtest.avi, progress bar at 15.7%, frames kept/crops counters
* `docs/screenshots/02_index_done.png` - indexing summary (time, motion-gate skips, crops, providers)
* `docs/screenshots/03_search_red_jacket.png` - "a woman in a red jacket": results grid with detection boxes + player
* `docs/screenshots/04_search_couple_on_grass.png` - "a man and a woman walking together on the grass": clicked result playing from 00:54
* `docs/screenshots/05_runtime_providers.png` - Runtime tab: EP per model (CPU here), latencies, ORT node placement
* `docs/screenshots/06_audit_log.png` - audit log of the queries

## 7. Three-minute demo script (only things that work today, on a CPU laptop)

Prep: `python -m footagefind index data/videos/vtest.avi data/videos/indoor_desk.avi` beforehand (about 80 s on the
4-vCPU VM), then `python -m footagefind.app` and open http://127.0.0.1:7860.

1. **0:00-0:25 Problem.** "After an incident, a shop owner or society secretary scrubs hours of DVR export by hand.
   The footage is private, so it shouldn't go to the cloud." Show the header line: offline, no face recognition, audit log.
2. **0:25-0:55 Index.** Index tab -> pick `data/videos/indoor_desk.avi` -> Index video. Point at the progress bar and
   the summary (frames sampled, motion-gate skips, crops, which execution provider ran each model). Say that
   vtest.avi was indexed in advance (it takes ~80 s on this CPU).
3. **0:55-1:45 Search.** Search tab. Type "a woman in a red jacket" -> results grid with yellow detection boxes.
   Click a result; the video plays from 1 s before that moment. Then "a man and a woman walking together on the
   grass" -> click the top result (00:55). Then "a man holding a colourful ball" -> the top result comes from the
   *other* video (indoor clip), showing multi-camera search.
4. **1:45-2:15 Why it is accurate.** Show the README results table: crops raise Recall@1 from 0.56 to 0.94 on 16
   hand-labelled queries. The w8a16 NPU-format models keep the same Recall@1 (0.94), while naive INT8 drops to 0.75.
   State clearly that these are CPU measurements.
5. **2:15-2:40 Snapdragon path.** Runtime tab: one ONNX Runtime code path; on a Snapdragon X laptop the same app
   with `--config configs/snapdragon.toml` requests QNNExecutionProvider on the Hexagon NPU with CPU fallback
   disabled. Say honestly that NPU numbers are the next step (AI Hub script ready, not run).
6. **2:40-3:00 Responsibility.** Audit log tab: every search is recorded locally. No faces, no identities,
   no cloud. Close.

Do not demo the "Verify with local VLM" checkbox unless a local OpenAI-compatible VLM server is running; it is untested.
