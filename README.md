# Real-Time Computer Vision & ONNX Inference Pipeline

Equivalent object-detection pipelines in modern C++ and Python using OpenCV and ONNX Runtime on Ubuntu Linux. The finished project will make preprocessing, inference, postprocessing, rendering, and benchmarking directly comparable across both implementations.

> Status: Phase 1 Python media passthrough is implemented. Images, videos, and cameras can be displayed or saved without modifying frames. C++ remains a Phase 0 help/version bootstrap. ONNX inference and benchmarks are not implemented.

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

Run all Phase 0 checks:

```bash
PYTHON_BIN=python3.11 scripts/smoke_test.sh
```

## Python Media Pipeline (Phase 1)

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

Without `--output`, frames are only displayed (or read and discarded with `--no-display`). Q, ESC, or Ctrl+C exits; an interactive image stays visible until exit. Videos stop at EOF; cameras run until exit or `--max-frames`. The frame limit must be a positive integer. `--confidence` (default 0.25) and `--iou` (default 0.45) accept finite numbers in [0, 1] and have no effect in Phase 1.

Image inputs support PNG, JPEG, BMP, TIFF, WebP, and PNM (`.ppm`, `.pgm`, `.pbm`). Video inputs support `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.mpg`, `.mpeg`, and `.wmv`, subject to installed decoders. URLs, directories, and other extensions are rejected. Files need a supported extension, including numeric names such as `0.png`.

Image sources save to image formats; video/camera sources save to `.avi`/`.mkv` (MJPG), `.mp4`/`.mov` (mp4v), or `.webm` (VP80) where the encoder is available. Video saving requires constant, even frame dimensions to avoid silent codec cropping. The source FPS is used when reported, otherwise 30 FPS. Video/image encoding may be lossy depending on the chosen format; no detections, overlays, or image transformations are applied before encoding.

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

YOLOX-Nano is the planned default detector, but no model weights are committed. The exact artifact URL, checksum, tensor contract, and redistribution notes will be audited in Phase 2 and documented in `docs/MODEL.md`.

The license for this repository's original code has not been selected yet. A `LICENSE` file will be added only after the repository owner confirms the preference. Third-party models and assets will retain separate provenance and license notes.
