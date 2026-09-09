# Model and class labels

Phase 2 uses the official Megvii YOLOX-Nano ONNX asset from release `0.1.1rc0`:

- [Human-readable download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx).
- [Pinned official asset ID 42724905](https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/assets/42724905), downloaded with `Accept: application/octet-stream`.
- Size: **3,659,407 bytes**.
- Computed SHA-256: **`c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`**.

```bash
python/.venv/bin/python scripts/download_or_export_model.py
python/.venv/bin/python scripts/download_or_export_model.py --verify-only
```

The helper is idempotent and uses only Python 3.11+ standard-library code. It downloads to a temporary file, verifies size/hash, and atomically publishes `models/detector.onnx`. A failed download never replaces the destination. `--output /path/to/detector.onnx` selects another location. The old GitHub release is not marked immutable; the asset ID and hash pin the accepted bytes. No `latest` URL is used.

`detector.onnx` is ignored and must not be committed. `classes.txt` contains the 80 ordered COCO names from [Megvii's coco_classes.py at 6ddff4824372906469a7fae2dc3206c7aa4bbaee](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/data/datasets/coco_classes.py), transformed from a Python tuple to one label per line. Copyright (c) Megvii, Inc. and its affiliates. Its Apache-2.0 source license is retained in [LICENSE.YOLOX](LICENSE.YOLOX); this does not select a license for original repository code.

The source repository is Apache-2.0, but the release artifact has no separate weight license/model card or embedded license metadata. No additional weight/data redistribution rights are asserted. No third-party images are included. See [the full audit](../docs/MODEL.md) for graph evidence, upstream provenance, licensing limits, and the exact BGR/0..255 preprocessing and raw-output decoding contract.
