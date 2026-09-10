# Audited YOLOX-Nano contract

This is the fixed model/math contract used by the completed Phase 0–8 core. Phase 6 parity and Phase 7 benchmarking do not expand this contract.

Audit date: 2026-09-09. The actual downloaded bytes were checked with ONNX 1.22.0 (`onnx.checker.check_model`) and executed with ONNX Runtime 1.24.4 using only `CPUExecutionProvider`, Python 3.12.14, OpenCV 4.14.0, and NumPy 2.5.3.

## Artifact identity

- Official repository: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
- Release: [0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0), release ID 48035658.
- Candidate/download link: <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx>.
- Pinned asset identity: [42724905](https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/assets/42724905), requested with `Accept: application/octet-stream`. Both URLs were downloaded independently and yielded identical hashes. The helper uses the asset-ID URL so replacing a same-named release attachment cannot select a different asset ID.
- Size: **3,659,407 bytes**.
- SHA-256 computed from the downloaded file: **`c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`**.
- GitHub reports this old release as `immutable: false`, and supplies no upstream digest. The numeric asset ID and locally audited SHA-256 pin the accepted content; the tag URL alone is not an immutability guarantee. Availability still depends on upstream retaining the asset.
- Local destination: `models/detector.onnx`, ignored by Git. No weights are redistributed in this repository.

## Inspected graph and tensors

| Property | Observed value |
|---|---|
| ONNX opset / IR | `ai.onnx` 11 / IR 6 |
| Producer | PyTorch 1.7 |
| Model metadata properties | None (including no embedded license) |
| Input | `images`, `tensor(float)`, `[1, 3, 416, 416]` |
| Output | `output`, `tensor(float)`, `[1, 3549, 85]` |
| Execution provider | `CPUExecutionProvider` |
| Output rows | `52*52 + 26*26 + 13*13 = 3549`, strides 8, 16, 32 |
| Row fields | raw `tx, ty, tw, th`, objectness probability, 80 class probabilities |

The output branches concatenate regression Conv outputs with Sigmoid objectness and class outputs, then Reshape/Concat/Transpose. There is **no Exp operator** and no grid/stride decode tail. The first operators slice the BGR tensor into the Focus stem; there is no input color conversion or normalization in the graph. A constant-114 CPU input produced finite float32 output of the stated shape, including negative raw box values. No additional sigmoid or softmax belongs in Python.

The runtime accepts only the audited SHA-256 and exact static tensor metadata. Names are read from the session. This intentionally rejects unreviewed exports, even exports with deceptively matching shapes (for example, a graph with embedded decoding). Other model families, dynamic sizes/batches, GPU providers, and alternate exports are unsupported.

## Upstream evidence and preprocessing

Reviewed upstream revision: **`6ddff4824372906469a7fae2dc3206c7aa4bbaee`**. Immutable source links:

- [Official ONNX deployment README](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/demo/ONNXRuntime/README.md) explicitly links the audited Nano artifact and specifies 416x416.
- [ONNX inference demo](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/demo/ONNXRuntime/onnx_inference.py).
- [`preproc` in data_augment.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/data/data_augment.py).
- [Decode and class-aware NMS](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/utils/demo_utils.py).
- [Detection head](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/models/yolo_head.py) and [exporter](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/tools/export_onnx.py).

The release-tag snapshot `0.1.1rc0` still contains the legacy RGB/255/mean/std preprocessing. It is not the deployment source used here: the maintained deployment README explicitly maps this artifact to the non-legacy demo above, and the release notes explain the removal of normalization and legacy weight incompatibility. Tensor metadata alone cannot prove a learned color contract; the linked deployment implementation is the evidence for BGR/raw-value preprocessing. No training/export reproduction is claimed.

For an HxWx3 uint8 OpenCV BGR frame:

1. Compute `r = min(416 / H, 416 / W)`.
2. Resize to `(int(W*r), int(H*r))` using OpenCV `INTER_LINEAR`, with uint8 pixel rounding.
3. Place the resized image at the top-left of a 416x416 uint8 BGR canvas filled with 114. Padding is on the right/bottom only.
4. Transpose HWC to CHW, make contiguous float32, and add batch dimension: `[1,3,416,416]`.
5. **Keep BGR channel order and 0..255 values. No `/255`, RGB conversion, mean/std normalization, or centered letterbox.**

Metadata records original/model/resized `(height, width)`, the unrounded scalar ratio, and `(left, top, right, bottom)` padding. Inversion uses the scalar ratio, as upstream does, rather than independently rounded x/y ratios. Empty, non-uint8, non-three-channel frames and extreme aspect ratios that round one resized dimension to zero are rejected.

## Decoding and selection

- Rows are concatenated by stride 8, 16, 32; each level is row-major in `(y,x)` grid order.
- Decode once outside the graph: `cx=(tx+grid_x)*stride`, `cy=(ty+grid_y)*stride`, `w=exp(tw)*stride`, `h=exp(th)*stride`.
- Convert center/size to corners. Undo padding and divide by the original scalar ratio.
- Each class score is `objectness * class_probability`. No logits conversion is applied.
- Use class-aware, multi-label selection: one anchor can yield multiple classes. The upstream demo defaults to class-agnostic selection; this project explicitly uses its class-aware semantics as required by Phase 2.
- Keep scores **greater than or equal to** `--confidence`. This inclusive boundary follows the project's “remove below threshold” requirement; the upstream helper uses strict `>`.
- NMS runs separately per class on restored, unclamped coordinates, using the upstream pixel-inclusive `+1` area/intersection convention. Suppress only when IoU is **greater than** `--iou`; equality survives. Applying NMS after coordinate restoration matters because of this `+1` convention.
- Ties within a class keep the lower original anchor index. Results are globally ordered by descending confidence, then ascending class ID and anchor index. This makes ties deterministic rather than inheriting an unstable sort.
- After NMS, clamp continuous box edges to `[0,W]` and `[0,H]`, and discard boxes with no remaining area. The renderer clamps raster coordinates further to `[0,W-1]` and `[0,H-1]`.
- Reject wrong ranks/shapes, a class count other than 80, non-float32 output, non-finite data, invalid probabilities, and decode overflow with explicit errors. No detections is a successful result.

## Labels, license, and redistribution

`models/classes.txt` is the ordered 80-name tuple from [coco_classes.py at the reviewed revision](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/data/datasets/coco_classes.py), converted to one name per line. IDs are contiguous zero-based indices, not sparse COCO annotation category IDs. Copyright (c) Megvii, Inc. and its affiliates. The source is distributed under the repository's [Apache-2.0 license](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/LICENSE); a copy is retained at `models/LICENSE.YOLOX`. The preprocessing/decode/NMS implementations follow those attributed upstream algorithms, with the project-specific boundaries and ordering noted above.

The official model release has no separate model license/card or embedded license metadata in the downloaded graph. The source repository is Apache-2.0; that is not independent evidence of distinct weight/data redistribution permissions. We link to and locally verify the official artifact, do not commit its bytes, and do not claim additional weight or training-image rights. No COCO images or annotations are included. Original project code still has no selected repository license.

Custom `--labels` files must be UTF-8 with exactly 80 nonempty unique lines in the same semantic order; changing display names cannot change the model's learned classes. Omitting `--labels` with `--model` uses the repository's `models/classes.txt`.

## Acquisition and checks

### C++ runtime archive

Phase 5 uses the official CPU-only Linux x86-64 ONNX Runtime **1.24.4** archive: `https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz`. The audited archive is **8,155,822 bytes** with SHA-256 **`3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747`**. `scripts/setup_cpp.sh` downloads it atomically into ignored `third_party/onnxruntime/`, verifies it before publication, and validates `include/onnxruntime_cxx_api.h` plus `lib/libonnxruntime.so`.

```bash
python/.venv/bin/python scripts/download_or_export_model.py
python/.venv/bin/python scripts/download_or_export_model.py --verify-only
python/.venv/bin/python -c "import onnxruntime as ort; s=ort.InferenceSession('models/detector.onnx', providers=['CPUExecutionProvider']); print([(i.name,i.shape,i.type) for i in s.get_inputs()]); print([(i.name,i.shape,i.type) for i in s.get_outputs()])"
```

The helper verifies existing files without downloading again, stages downloads beside the destination, verifies size and SHA-256, and atomically replaces only verified content. `--verify-only` never downloads or alters the destination. Errors retain any existing destination and clean up the temporary download. The optional `model-tools` Python dependency group supplies `onnx` for inspection only; ONNX Runtime alone is used for inference.

## Phase 2 validation evidence

The pinned upstream `preproc` and `demo_postprocess` functions were run offline for comparison. Six deterministic portrait/landscape/square/small-image tensors and scale ratios matched exactly, with no tolerance. All 3,549 decoded boxes from the actual CPU model output matched exactly. The three class-aware detections on the generated gradient at confidence 0.01 also matched exactly after applying the documented project clipping and ordering. Threshold equality and tie behavior are separately specified and tested rather than hidden behind numerical tolerances.

The real-model smoke test uses a generated 240x320 BGR gradient, runs CPU inference, and saves a readable annotated PNG. The low threshold deliberately exercises drawing on synthetic data; those predictions are not evidence of real-world detection accuracy. Automated GUI/camera behavior is mocked. Video writer availability is OpenCV/backend-dependent.
