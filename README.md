# Real-Time Computer Vision & ONNX Inference Pipeline

Equivalent object-detection pipelines in modern C++ and Python using OpenCV and ONNX Runtime on Ubuntu Linux. The finished project will make preprocessing, inference, postprocessing, rendering, and benchmarking directly comparable across both implementations.

> Status: Phase 7 adds compatible Python/C++ CPU benchmark export and one validated, host-specific comparison. Phase 8 has not started.

## Planned Stack

- C++17, CMake 3.20+, OpenCV 4, ONNX Runtime C++ API
- Python 3.11+, OpenCV, NumPy, ONNX Runtime, pytest, Ruff
- CPU-first inference and reproducible cross-language benchmarks

## Planned Runtime Flow

```text
Input -> Frame capture -> Preprocess -> ONNX Runtime -> Postprocess/NMS
      -> Render -> Display/save -> Metrics and benchmark export
```

Python and C++ will use the same model artifact, thresholds, resize policy, tensor layout, output decoding, NMS rules, and timing definitions.

## Bootstrap Checks

Python help without installing runtime dependencies:

```bash
PYTHONPATH=python/src python3.11 -m vision_pipeline --help
```

C++ help-only target:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vision_cpp --help
ctest --test-dir build --output-on-failure
```

## C++ Media Pipeline (Phase 4)

The C++ executable now passes original frames through without loading a model or drawing detections. Build it as above, then use local paths:

```bash
# Headless image passthrough; parent directories are created.
./build/vision_cpp --source path/to/image.png --output outputs/copy.png --no-display

# Headless video passthrough, limited to 100 frames.
./build/vision_cpp --source path/to/video.mp4 --output outputs/copy.avi --no-display --max-frames 100

# Interactive camera input; use Q or ESC to exit.
./build/vision_cpp --source 0
```

`--source` accepts a bare non-negative camera index or a supported local image/video file. `--confidence` and `--iou` are validated as finite values in [0, 1] and control detection when `--model` is supplied. `--max-frames` must be positive. Use `--no-display` on headless systems. Image sources save image formats; video and camera sources save `.avi`/`.mkv` (MJPG), `.mp4`/`.mov` (mp4v), or `.webm` (VP80) when the local OpenCV codec is available. Output is staged beside the requested destination and decoded before it replaces an existing file. CTest uses a small self-contained assertion executable because GoogleTest is not installed in this environment; it downloads no test dependencies.

For Phase 5, install the pinned local ONNX Runtime CPU archive, configure CMake with that path, and supply the audited model to enable labeled boxes:

```bash
scripts/setup_cpp.sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime"
cmake --build build -j
./build/vision_cpp --model models/detector.onnx --labels models/classes.txt \
  --source path/to/image.png --output outputs/detected.png --no-display
```

The C++ detector uses one CPU ONNX Runtime session, BGR 0..255 top-left letterboxing, raw YOLOX decoding, class-aware NMS, and labelled rendering. It accepts only the audited static `[1,3,416,416]` to `[1,3549,85]` float32 contract. Phase 6 parity commands and limits are documented in [docs/PARITY.md](docs/PARITY.md).

Run all Phase 0 checks:

```bash
PYTHON_BIN=python3.11 scripts/smoke_test.sh
```

## Python ONNX Detector (Phase 2)

Set up Python 3.11+ with `PYTHON_BIN=python3.11 scripts/setup_python.sh`, then download and verify the official model from the repository root:

```bash
python/.venv/bin/python scripts/download_or_export_model.py
python/.venv/bin/python scripts/download_or_export_model.py --verify-only

# Image inference, saved headlessly.
python/.venv/bin/python -m vision_pipeline --model models/detector.onnx --labels models/classes.txt --source path/to/image.png --output outputs/detected.png --no-display

# Video inference, stopping after 100 frames.
python/.venv/bin/python -m vision_pipeline --model models/detector.onnx --source path/to/video.mp4 --output outputs/detected.avi --no-display --max-frames 100

# Interactive camera inference.
python/.venv/bin/python -m vision_pipeline --model models/detector.onnx --source 0 --confidence 0.25 --iou 0.45
```

`--model` enables detection. `--labels` is optional and defaults to this repository's `models/classes.txt`; custom files need exactly 80 unique, nonempty UTF-8 names in the same COCO order. Labels cannot be supplied without a model. The model and labels are validated before opening media; one CPU session is reused throughout the run. Missing/unreadable paths are configuration errors (exit 2), while bad checksums, incompatible models, and inference/media failures return 1 with context and no traceback. Omitting `--model` preserves the original passthrough commands below.

The pinned model is official YOLOX-Nano `0.1.1rc0`, **3,659,407 bytes**, SHA-256 **`c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`**. Its opset-11 tensors are `images: float32[1,3,416,416]` and `output: float32[1,3549,85]`. Preprocessing retains **BGR 0..255 values**, uses bilinear resize and top-left padding of 114, and produces contiguous NCHW float32. There is no RGB conversion or normalization. Grid/stride decoding is outside the graph. Scores are objectness times class probability; NMS is class-aware with deterministic ties and the audited pixel-inclusive IoU convention. Boxes are restored and clamped before drawing class labels/confidence on a frame copy. No detections is a valid successful result; no FPS/latency overlay is added.

Only the audited model bytes and static batch-one 416x416 contract are supported. The helper pins an official numeric GitHub asset ID plus SHA-256: the old release itself is not marked immutable. See [docs/MODEL.md](docs/MODEL.md) for the full audit, exact threshold/NMS boundaries, source revisions, and licensing limitations. The code repository is Apache-2.0 upstream, but the released weights have no separate license/card or embedded license metadata; weights are downloaded locally and remain ignored. The attributed class list and upstream license are under `models/`.

Tested with Python 3.12.14, ONNX Runtime 1.24.4 (`CPUExecutionProvider`), OpenCV 4.14.0, and NumPy 2.5.3. CPU inference requires no network after acquisition. The optional inspection dependency can be installed with `python/.venv/bin/python -m pip install -e './python[dev,model-tools]'`; `onnx` is not a runtime dependency. GUI/camera checks are mocked, and video saving depends on installed codecs.

```bash
python/.venv/bin/ruff format python scripts
python/.venv/bin/ruff check python scripts
python/.venv/bin/ruff format --check python scripts
python/.venv/bin/pytest python/tests
python/.venv/bin/pytest python/tests/test_smoke.py -rs
python/.venv/bin/python -m pip check
```

Tests never download weights automatically. The real-model test skips only if the local model is absent and fails if present but invalid. To run only unit tests, use `python/.venv/bin/pytest python/tests -m "not real_model"`. The real-model smoke generates a deterministic color-gradient image and saves a readable annotated PNG at an intentionally low confidence threshold of 0.01. This exercises the full detector and renderer; synthetic-image predictions do not establish detection accuracy.

## Comparative metrics and benchmark mode (Phase 7)

```bash
python/.venv/bin/python -m vision_pipeline \
  --model models/detector.onnx --labels models/classes.txt \
  --source path/to/input.avi --benchmark --no-display \
  --warmup 5 --max-frames 200 \
  --benchmark-output benchmarks/results/python-run.json
```

Both CLIs require `--no-display` for benchmarking and write the requested JSON plus sibling CSV after a clean run. `--warmup` defaults to 5 and `--max-frames` limits measured frames, not warm-up frames. Metrics cover capture, preprocessing, inference, postprocessing, rendering, total processing time, effective FPS, and an approximate RSS snapshot; GUI wait time is excluded. Use [benchmarks/run_benchmarks.py](benchmarks/run_benchmarks.py) with the tracked `assets/sample/benchmark.avi` fixture to validate compatible real runs. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) and the measured [benchmark report](benchmarks/REPORT.md).

## Python Media Pipeline (Passthrough)

Set up with an installed Python 3.11+ interpreter:

```bash
PYTHON_BIN=python3.11 scripts/setup_python.sh
python/.venv/bin/python -m vision_pipeline --help
python/.venv/bin/python -m vision_pipeline --version
```

Replace the example input paths with your own local files:

```bash
# Headless image passthrough; missing output directories are created.
python/.venv/bin/python -m vision_pipeline --source path/to/image.png --output outputs/copy.png --no-display

# Headless video passthrough, limited to 100 frames.
python/.venv/bin/python -m vision_pipeline --source path/to/video.mp4 --output outputs/copy.avi --no-display --max-frames 100

# Interactive webcam (bare non-negative integers select cameras).
python/.venv/bin/python -m vision_pipeline --source 0

# Headless webcam recording with an explicit stopping point.
python/.venv/bin/python -m vision_pipeline --source 0 --output outputs/camera.avi --no-display --max-frames 100
```

Without `--output`, frames are only displayed (or read and discarded with `--no-display`). Q, ESC, or Ctrl+C exits; an interactive image stays visible until exit. Videos stop at EOF; cameras run until exit or `--max-frames`. The frame limit must be a positive integer. `--confidence` (default 0.25) and `--iou` (default 0.45) accept finite numbers in [0, 1] and control detection when `--model` is supplied; they have no effect in passthrough mode.

Image inputs support PNG, JPEG, BMP, TIFF, WebP, and PNM (`.ppm`, `.pgm`, `.pbm`). Video inputs support `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.mpg`, `.mpeg`, and `.wmv`, subject to installed decoders. URLs, directories, and other extensions are rejected. Files need a supported extension, including numeric names such as `0.png`.

Image sources save to image formats; video/camera sources save to `.avi`/`.mkv` (MJPG), `.mp4`/`.mov` (mp4v), or `.webm` (VP80) where the encoder is available. Video saving requires constant, even frame dimensions to avoid silent codec cropping. The source FPS is used when reported, otherwise 30 FPS. Video/image encoding may be lossy depending on the chosen format; passthrough frames are unchanged, while detector runs save the annotated frames.

Output is staged beside the destination and decoded for verification before replacing it. Video verification reads all saved frames, adding time at shutdown. A failed run leaves existing output intact and reports an error; a successful Ctrl+C finalizes completed frames and returns exit status 130. Configuration errors return 2 and media errors return 1. Output cannot alias the source file. GUI use requires a working desktop session and an OpenCV GUI backend; use `--no-display` on servers. Camera and GUI behavior are mocked in automated tests.

Run the Python checks (fixtures are generated in pytest temporary directories):

```bash
python/.venv/bin/ruff check python
python/.venv/bin/ruff format --check python
python/.venv/bin/pytest python/tests
```

## Development Plan

See [PLAN.md](PLAN.md) for the dependency-ordered phases, task-level verification commands, acceptance criteria, model strategy, and current environment risks.

## Model and Licensing Status

YOLOX-Nano is the audited Phase 2 detector. No model weights are committed. The artifact URL, checksum, tensor contract, source/license provenance, and redistribution limitations are documented in [docs/MODEL.md](docs/MODEL.md).

The license for this repository's original code has not been selected yet. A `LICENSE` file will be added only after the repository owner confirms the preference. Third-party models and assets will retain separate provenance and license notes.
