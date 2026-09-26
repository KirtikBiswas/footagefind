# Running FootageFind on a Snapdragon X laptop (Hexagon NPU)

> **Status: untested on device.** Everything in this document is the plan and the
> code path that was built and unit-tested on an x86 Linux VM. No part of it has
> been run on Snapdragon hardware, through QNN, or through Qualcomm AI Hub yet.
> No NPU performance numbers exist for this project; see the empty table in
> [`results/RESULTS.md`](../results/RESULTS.md).

## 1. What stays the same

The application code does not change between an x86 dev box and a Snapdragon X
laptop. All three models (CLIP ViT-B/32 image encoder, CLIP text encoder,
YOLOv8n) run through `footagefind/runtime.py::OrtModel`, which builds an ONNX
Runtime session from the provider list in the config:

```toml
# configs/snapdragon.toml
[runtime]
providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
qnn_backend_path = "QnnHtp.dll"
disable_cpu_ep_fallback = true
qnn_context_cache = true
```

```powershell
python -m footagefind.app --config configs/snapdragon.toml
python -m footagefind --config configs/snapdragon.toml runtime   # which EP each model loaded on
```

PyTorch is only used by `scripts/export_onnx.py`. The tokenizer is vendored
(`footagefind/tokenizer.py`), so the device needs only numpy, OpenCV,
onnxruntime-qnn and Gradio.

## 2. What changes on device

| Area | x86 dev box (what was tested) | Snapdragon X laptop (planned) |
|---|---|---|
| Python | CPython 3.11 x86-64 | **Native ARM64** CPython (python.org "Windows installer (ARM64)"). An x64 Python under emulation cannot load the ARM64 QNN libraries. |
| ONNX Runtime | `onnxruntime` (CPU EP) | **`onnxruntime-qnn`** (ships `QnnHtp.dll`, the HTP stub and skel libraries). Uninstall plain `onnxruntime` first; both install the same `onnxruntime` module. |
| Execution provider | `CPUExecutionProvider` | `QNNExecutionProvider` with `backend_path=QnnHtp.dll`, `htp_performance_mode=burst`, `enable_htp_fp16_precision=1` (set in `runtime.qnn_provider_options`) |
| Model precision | FP32 ONNX | **w8a16 QDQ ONNX** (INT8 weights, UINT16 activations): `models/onnx/*.w8a16.onnx` from `scripts/quantize_onnx.py`, or an AI Hub quantize job. FP32 graphs also run (as FP16 on the HTP) but are larger and slower. |
| Shapes | static | **static, and must stay static**: image `[1,3,224,224]` float32, tokens `[1,77]` **int32**, YOLO `[1,3,640,640]` float32. The HTP compiles a graph for fixed shapes; a dynamic dimension means a failed compile or CPU fallback. A different batch size needs a re-export (`--image-batch N`). `ClipEmbedder._run_batched` pads the last partial batch. |
| First load | ~1 s per model | Graph finalisation for the HTP can take a long time on first load. `runtime.py` sets `ep.context_enable=1` and writes a **QNN context binary** next to the model (`<model>_ctx.onnx` + `.bin`, `ep.context_embed_mode=0`); later launches load that directly. Delete the `_ctx` files after changing the model, onnxruntime-qnn or the NPU driver. AI Hub's `--target_runtime precompiled_qnn_onnx` produces an equivalent pre-compiled artefact. |
| CPU fallback | n/a | `disable_cpu_ep_fallback = true` adds `session.disable_cpu_ep_fallback=1`. ORT then **refuses to create the session** if any node is unsupported by QNN, instead of silently running it on the CPU. ORT 1.30 also rejects the CPU EP being registered at the same time, so `select_providers()` drops it in strict mode (covered by `tests/test_runtime.py`). |
| Threads | decode thread + inference thread | Same code (`pipeline.py`): OpenCV decoding and motion gating run on the `ff-decode` thread, feeding a bounded queue (8 frames); YOLO and CLIP run on the main thread. Both release the GIL, so the Oryon CPU cores decode while the NPU infers. |
| VLM verification | client + stub-server tests only | Qwen3-VL-4B-Instruct served by Qualcomm GenieX (`geniex serve`, OpenAI-compatible HTTP on localhost). **Lazy**: nothing VLM-related is loaded unless the user ticks "Verify". Set `verify.start_command` to your GenieX launch command and `ensure_server()` starts it on first use, so the VLM's memory is not held while indexing or doing CLIP-only search. |

### Why w8a16 and not plain INT8

Measured on the x86 VM (CPU numerics are a faithful preview of what a QDQ graph
computes, even though CPU speed says nothing about NPU speed): mean cosine
similarity of quantized vs FP32 CLIP embeddings.

| Encoder | INT8 w8a8 (MatMul/Gemm/Conv only) | w8a16 (ORT QNN recipe, all ops) |
|---|---|---|
| CLIP image | 0.846 | 0.999 |
| CLIP text | 0.789 | 0.999 |

Retrieval on the 16 eval queries (CPU, same index settings): FP32 R@1 0.94, w8a8 0.75, w8a16 0.94, and
all-w8a16 including YOLO 0.88. YOLOv8n w8a16 matched 301/301 FP32 detections at IoU >= 0.5 on 40 frames.
Source: `results/RESULTS.md`.

In the calibration sweep on the image encoder, 8-bit activations lost a lot of
fidelity (cosine about 0.76-0.85 depending on the calibration method), while 16-bit
activations kept about 0.998. Transformers have outlier activations that 8 bits
cannot cover. The retrieval accuracy of each variant is in the RESULTS table.

### Two pitfalls that were found and fixed

open_clip's text encoder uses a causal attention mask of `0 / -inf`. `-inf`
breaks MinMax calibration, and the first w8a16 text encoder came out with a
cosine of 0.37 to FP32 (useless). `scripts/export_onnx.py::finite_attention_masks`
rewrites the mask to `0 / -100` (identical after softmax; FP32 max-abs-diff to
PyTorch stays ~2e-7). After that, w8a16 text cosine is ~0.9995.

The QNN QDQ config also quantizes YOLOv8's `Resize` **scales** input. `[1,1,2,2]` in UINT16
dequantizes to `0.99998`, and ORT computes `floor(256 x 0.99998) = 255` channels, so the neck silently lost a
feature map and w8a16 YOLO kept only 12% of FP32 detections. `scripts/quantize_onnx.py::resize_scales_to_sizes`
rewrites Resize to use integer `sizes` before quantizing (regression test in `tests/test_runtime.py`). If you
use AI Hub's quantizer instead, check the same thing.

### Known risks on device (not yet checked)

* **Op coverage.** The text encoder ends with `ArgMax` over token ids plus a
  `Gather` to pool the end-of-text token. If the HTP rejects either, strict mode
  will fail at session creation. The fallback is to compute the EOT index in
  numpy (it is the position of the largest token id) and export an encoder that
  takes it as a second input.
* **YOLO head.** The Ultralytics head's decode (`Sigmoid`, `Concat`, `Split`,
  `Mul` on anchors) is usually HTP-friendly, but w8a16 quantization of box
  coordinates may shift boxes; `eval/run_eval.py` reports w8a16-vs-FP32
  detection agreement on CPU as a preview.
* **Wheels.** `opencv-python-headless` and Gradio's compiled dependencies need
  win_arm64 wheels for the chosen Python version. Check before a demo.

## 3. Setup on the laptop (Windows 11 on ARM)

```powershell
# 1. Native ARM64 Python 3.11 or 3.12 from python.org, then:
py -3.11-arm64 -m venv .venv ; .venv\Scripts\activate
pip install -r requirements-snapdragon.txt
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
#   expect: ['QNNExecutionProvider', 'CPUExecutionProvider']

# 2. Copy models/onnx/ from the x86 dev box (FP32 + *.w8a16.onnx), or produce the
#    w8a16 models with AI Hub (scripts/aihub_compile_profile.py --precision w8a16).
python scripts/download_assets.py --videos

# 3. Prove full-NPU placement, then run
python -m footagefind --config configs/snapdragon.toml runtime
python -m footagefind --config configs/snapdragon.toml index data/videos/vtest.avi
python -m footagefind.app --config configs/snapdragon.toml
```

## 4. How to verify the NPU is actually used

1. **Strict mode.** With `disable_cpu_ep_fallback = true` the session either
   loads fully on QNN or raises. `python -m footagefind --config configs/snapdragon.toml runtime`
   prints `primary_provider` per model; the app's **Runtime** tab shows the same
   ("Running on: NPU (QNN HTP)").
2. **Node placement from ORT profiling.** The Runtime tab's "Probe node
   placement" button (and the `runtime` CLI command) runs each model once with
   `enable_profiling=True` and counts kernel events per provider. A fully
   offloaded model shows one or a few fused `QNNExecutionProvider` nodes and no
   `CPUExecutionProvider` nodes.
3. **Task Manager.** Performance tab -> **NPU** graph. Index a video and watch
   NPU utilisation rise while CPU stays moderate (the CPU still decodes video).
4. **QNN profiling** (optional): add `profiling_level=detailed` and
   `profiling_file_path=qnn_profile.csv` to `qnn_provider_options()` for
   per-op HTP timings.
5. **ORT verbose log**: `so.log_severity_level = 0` prints the partitioning
   ("Number of partitions supported by QNN EP").

## 5. What to measure on device (fills the empty table)

```powershell
python scripts/aihub_compile_profile.py --precision fp32 w8a16 --check-numerics   # needs AI Hub token
python eval/run_eval.py --quick   # after pointing it at configs/snapdragon.toml (see HANDOFF.md)
```

Record: per-model mean/p95 latency on QNN, peak memory, indexing frames/s,
query latency, Recall@k on the same 16 queries, and share of nodes on QNN.
Never copy a number into the README that was not produced by one of these runs.
