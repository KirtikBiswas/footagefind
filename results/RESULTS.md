# FootageFind - measured results

All numbers below were measured by `python eval/run_eval.py` on **Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, 15.7 GiB RAM, Linux 6.18.44-fc-v37 (x86_64), onnxruntime 1.30.0 CPUExecutionProvider** (virtual machine), on 2026-09-26.
No number in this file was measured on, or extrapolated to, Snapdragon hardware.

Test data: `vtest.avi` (79.5 s, 768x576, OpenCV sample) + `indoor_desk.avi` (6.6 s, 1920x1080, opencv_extra) indexed together; 16 queries with manually labelled ground truth (`eval/queries.json`), hit tolerance +/-0.25s, sampling 2.0 fps, merge window 3.0 s. Ranks are computed over the full merged result list.

## Retrieval accuracy (all queries)

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

## Indexing cost

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

## Motion gating on a synthetic idle-padded clip

`vtest.avi` never has a static moment (people walk through every frame), so the gate keeps 100% of its frames. To show what the gate does on typical CCTV (long idle stretches) we built **a synthetic clip**: 60 s of the frozen first frame + vtest + 60 s of the frozen last frame, with Gaussian noise (sigma=2.0) added to the frozen parts. Ground truth is shifted by +60 s. Real DVR footage has compression artefacts, lighting changes and swaying trees, so real skip ratios will be lower than on this idealised clip.

| Configuration | Recall@1 | Recall@5 | Recall@10 | Median rank | Frames skipped | Index wall time (s) |
|---|---|---|---|---|---|---|
| Idle-padded vtest + indoor, crops, motion gate on | 0.875 | 0.938 | 1.00 | 1.00 | 236/413 | 80.0 |
| Idle-padded vtest + indoor, crops, motion gate off | 0.812 | 0.938 | 1.00 | 1.00 | 0/413 | 180.9 |

## Per-model latency (ONNX Runtime, CPUExecutionProvider)

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

## YOLOv8n w8a16 vs FP32 (CPU, detection agreement)

On 40 vtest frames: FP32 301 detections, w8a16 301, 301 matched at IoU>=0.5 with the same class (recall vs FP32 1.0, precision vs FP32 1.0).

## Quantized CLIP embedding fidelity (mean cosine to FP32 embeddings)

| Encoder | INT8 w8a8 | w8a16 |
|---|---|---|
| clip_image | 0.8458 | 0.9987 |
| clip_text | 0.7893 | 0.9994 |

## Per-query ranks

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

## Snapdragon X (AI Hub / on-device) - to be measured

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
