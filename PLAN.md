# Implementation Plan

This plan is derived from `CPP_Python_ONNX_Vision_Pipeline_Project_Specification.pdf` version 1.0. The PDF remains the source of truth. Work proceeds one phase at a time; a phase is not complete until its listed verification has actually run and its acceptance criteria are met.

## Working Rules

- Preserve existing valid work and keep changes scoped to the active phase.
- Use the same ONNX model file and equivalent image math in Python and C++.
- Use ONNX Runtime directly in both languages, with CPU execution first.
- Keep preprocessing and postprocessing pure and independently testable.
- Do not claim tests, builds, parity, or benchmarks that were not run.
- Record changed files, commands, results, and limitations at each phase boundary.
- Defer CUDA, FastAPI, cloud deployment, batching, asynchronous pipelines, and additional model families until the stable core meets the Definition of Done.

## Inspected Baseline

Repository state on 2026-09-08:

- The repository is an empty Git repository on branch `main`, with no commits and no tracked project files.
- Remote: `https://github.com/garikpetrosyan00/cpp-python-onnx-vision-pipeline.git`.
- Ubuntu toolchain found: Python 3.10.12, CMake 3.22.1, GCC/G++ 11.4.0, Git 2.34.1, and OpenCV C++ 4.5.4 through `pkg-config`.
- Clang is not installed.
- Python modules found: NumPy and pytest.
- Python modules not found: OpenCV, ONNX Runtime, psutil, and Ruff.
- ONNX Runtime C++ headers and libraries were not found in standard system locations.
- The specification requires Python 3.11+, so the current system Python cannot validate the final Python environment without installing or selecting a newer interpreter.

## Technical Decisions

### Default detector

Use the official YOLOX-Nano ONNX model as the first supported detector.

- Input target: `416x416`, batch size 1, float32 NCHW tensor.
- Runtime: ONNX Runtime `CPUExecutionProvider` in both implementations.
- Distribution: do not commit model weights. A pinned download helper will place the model at `models/detector.onnx`, verify a SHA-256 checksum, and record provenance.
- Labels: use a separately documented COCO class list at `models/classes.txt` only after its source and redistribution terms are recorded.
- Why: YOLOX-Nano is compact, the upstream project provides an ONNX Runtime path and a pre-generated ONNX artifact, and the upstream code is Apache-2.0 licensed. Its raw output keeps preprocessing, decoding, confidence filtering, NMS, and coordinate restoration visible for portfolio purposes.
- Upstream references: `https://github.com/Megvii-BaseDetection/YOLOX` and `https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime`.

This decision remains subject to a Phase 2 artifact audit: record the exact release URL, checksum, tensor names/shapes, opset, model-file licensing/provenance, and observed ONNX Runtime compatibility before treating the model contract as stable.

### ONNX Runtime versions and C++ setup

- Keep Python and C++ ONNX Runtime on the same pinned release line.
- Initial candidate: the ONNX Runtime 1.24 release line, because its Python support aligns with the specification's Python 3.11+ baseline. Phase 0 successfully installed the 1.24.4 Python wheel in a temporary Python 3.12 environment.
- Before pinning one exact version in both setup paths, verify that the matching official Linux x86-64 CPU archive works on Ubuntu 22.04 and this host's glibc.
- Install the C++ archive into a gitignored `third_party/onnxruntime/` cache through `scripts/setup_cpp.sh`; do not require a system-wide install.
- Expose `ONNXRUNTIME_ROOT` to CMake and fail configuration with an actionable message when it is missing or invalid.

### Repository license

Do not add a project `LICENSE` until the owner chooses the license for original repository code. Third-party model and asset notices remain separate from the repository's own license.

## Phase 0 - Planning and Repository Bootstrap

Goal: create a clean, buildable project shell without media processing or inference.

### 0.1 Repository skeleton

Files:

- Add `.gitignore`.
- Add the directory structure under `cpp/`, `python/`, `models/`, `assets/`, `benchmarks/`, `scripts/`, and `docs/`.
- Add `.gitkeep` only where an otherwise-empty directory must be retained.

Verification:

```bash
find . -maxdepth 4 -type f | sort
git status --short
```

Acceptance:

- The specification's top-level layout exists.
- Generated output, builds, virtual environments, downloaded models, local third-party binaries, caches, and temporary benchmark files are ignored.
- No model weight, generated media, binary dependency, or secret is committed.

### 0.2 Python package bootstrap

Files:

- Add `python/pyproject.toml` with Python 3.11+, runtime dependencies, pytest, and Ruff configuration.
- Add `python/src/vision_pipeline/__init__.py`, `__main__.py`, and a minimal `cli.py` that supports only `--help` and a version response.
- Add `python/tests/test_package.py` for import and CLI-help smoke coverage.

Verification:

```bash
python3.11 -m venv python/.venv
python/.venv/bin/pip install -e './python[dev]'
python/.venv/bin/ruff check python
python/.venv/bin/pytest python/tests
python/.venv/bin/python -m vision_pipeline --help
```

Acceptance:

- Package metadata installs under Python 3.11+.
- Import, help, lint, and bootstrap tests pass.
- No OpenCV or ONNX session is created in Phase 0.

### 0.3 C++ and CMake bootstrap

Files:

- Add `cpp/CMakeLists.txt`.
- Add `cpp/src/main.cpp` with help/version-only behavior.
- Add initial public include/source/test directories without inference code.

Verification:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vision_cpp --help
ctest --test-dir build --output-on-failure
```

Acceptance:

- CMake configures with CMake 3.20+ and C++17.
- The executable builds with `-Wall -Wextra -Wpedantic` on GCC/Clang.
- Help/version behavior works; no OpenCV capture or ONNX inference exists yet.

### 0.4 Documentation, setup placeholders, and CI skeleton

Files:

- Add base `README.md` with scope, planned architecture, current status, and bootstrap commands.
- Add `models/README.md` and `assets/README.md` with provenance/licensing rules.
- Add documentation skeletons in `docs/`.
- Add `scripts/setup_python.sh`, `scripts/setup_cpp.sh`, and `scripts/smoke_test.sh` as bootstrap-safe helpers.
- Add `.github/workflows/ci.yml` for Python 3.11 lint/tests and C++ configure/build/bootstrap tests.

Verification:

```bash
bash -n scripts/setup_python.sh scripts/setup_cpp.sh scripts/smoke_test.sh
python3 -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())"
```

Acceptance:

- Setup documentation distinguishes installed prerequisites from project-managed dependencies.
- CI contains no model download or inference claim.
- README clearly marks the project as Phase 0 rather than production-complete.

Phase 0 is complete only after all locally possible checks pass. Missing Python 3.11 or network-fetched dependencies must be reported explicitly rather than bypassed.

## Phase 1 - Python Media Pipeline

Goal: handle image, video, and camera sources without AI inference.

### 1.1 Configuration and source parsing

Files: `python/src/vision_pipeline/config.py`, `cli.py`, `input_source.py`, and `python/tests/test_config.py`.

Verification:

```bash
python/.venv/bin/pytest python/tests/test_config.py
python/.venv/bin/ruff check python
```

Acceptance: camera indices and file paths parse predictably; invalid thresholds, frame limits, and missing paths fail with actionable messages.

### 1.2 Frame acquisition and output lifecycle

Files: `input_source.py`, `render.py`, `pipeline.py`, `python/tests/test_input_source.py`, and small generated test fixtures.

Verification:

```bash
python/.venv/bin/pytest python/tests/test_input_source.py
python/.venv/bin/python -m vision_pipeline --source assets/sample/test.png --no-display --max-frames 1
```

Acceptance: image/video/camera semantics share one interface; EOF, Q/ESC, Ctrl+C, unavailable camera, headless mode, and output creation are handled cleanly.

## Phase 2 - Python ONNX Detector

Goal: run the audited YOLOX-Nano model end-to-end in Python.

### 2.1 Model artifact and contract audit

Files: `scripts/download_or_export_model.py`, `models/README.md`, `models/classes.txt`, and `docs/MODEL.md`.

Verification:

```bash
python scripts/download_or_export_model.py --verify-only
python -c "import onnxruntime as ort; s=ort.InferenceSession('models/detector.onnx', providers=['CPUExecutionProvider']); print([(i.name, i.shape, i.type) for i in s.get_inputs()]); print([(o.name, o.shape, o.type) for o in s.get_outputs()])"
```

Acceptance: source URL, release, SHA-256, licensing/provenance, input/output names and shapes, opset, preprocessing, and decoding contract are documented from the actual artifact.

### 2.2 Preprocessing

Files: `python/src/vision_pipeline/preprocess.py` and `python/tests/test_preprocess.py`.

Verification: `python/.venv/bin/pytest python/tests/test_preprocess.py`.

Acceptance: deterministic tests cover top-left letterbox scale/padding (value 114), unchanged BGR channel order and 0..255 values, float32 conversion without normalization, HWC-to-CHW, batch dimension, and coordinate metadata. Phase 2 artifact/upstream audit corrected the earlier RGB/normalization assumption; see `docs/MODEL.md`.

### 2.3 Session and postprocessing

Files: `inference.py`, `postprocess.py`, typed detection/config records, `test_postprocess.py`, and `test_smoke.py`.

Verification:

```bash
python/.venv/bin/pytest python/tests/test_postprocess.py python/tests/test_smoke.py
python/.venv/bin/python -m vision_pipeline --model models/detector.onnx --source assets/sample/test.png --no-display --max-frames 1
```

Acceptance: one session is created at startup; output decoding, confidence filtering, class-aware NMS, clamping, original-coordinate restoration, labels, and error messages are tested.

## Phase 3 - Python Metrics and Benchmark Mode

Goal: add defined timing boundaries and structured benchmark output.

Files: `python/src/vision_pipeline/metrics.py`, pipeline/CLI updates, metrics tests, and result schema documentation.

Verification:

```bash
python/.venv/bin/pytest python/tests
python/.venv/bin/python -m vision_pipeline --model models/detector.onnx --source assets/sample/demo.mp4 --benchmark --no-display --warmup 5 --max-frames 200
```

Acceptance: capture, preprocess, inference, postprocess, render, total, and FPS definitions match the specification; warm-up is excluded; count/mean/median/p50/p95/p99 and optional RSS are emitted to validated JSON/CSV.

## Phase 4 - C++ Media Pipeline

Goal: implement equivalent non-AI media and CLI behavior in modern C++.

Files: C++ config, input source, renderer, pipeline headers/sources; CMake OpenCV/test wiring; focused unit tests.

Verification:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vision_cpp --source assets/sample/test.png --no-display --max-frames 1
```

Acceptance: RAII-based image/video/camera handling, CLI validation, EOF/exit/output behavior, and errors match Python semantics where practical.

## Phase 5 - C++ ONNX Detector

Goal: run the same audited model with equivalent math through the ONNX Runtime C++ API.

Status: complete on the audited CPU-only path; Phase 6 parity and Phase 7 benchmarking remain unstarted.

Files: `preprocessor`, `inference_engine`, `postprocessor`, `detection`, and related tests; CMake ONNX Runtime discovery; `scripts/setup_cpp.sh` finalization.

Verification:

```bash
scripts/setup_cpp.sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime"
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vision_cpp --model models/detector.onnx --source assets/sample/test.png --no-display --max-frames 1
```

Acceptance: ONNX objects and buffers have clear RAII ownership; one session is reused; tensor conversion is explicit; tests cover preprocessing/postprocessing; unexpected model metadata fails with observed details.

## Phase 6 - Cross-Language Parity

Goal: prove Python and C++ produce equivalent canonical detections.

Status: complete for the audited CPU model and tracked 416x416 fixture. The real Python/C++ run produced 32 ordered detections within the documented `1e-5` absolute tolerance. Benchmarking remains Phase 7 work.

Files: JSON detection export in both CLIs, `scripts/check_parity.py`, fixed legal test image, and parity documentation/tests.

Verification:

```bash
scripts/smoke_test.sh
python scripts/check_parity.py --image assets/sample/parity.jpg --model models/detector.onnx
```

Acceptance: sorted class IDs match; confidence and original-frame box coordinates stay within documented tight tolerances; any difference is investigated and explained rather than hidden by broad thresholds.

## Phase 7 - Comparative Benchmark

Goal: measure both implementations under identical conditions.

Files: `benchmarks/run_benchmarks.py`, schema validation, selected result files, `benchmarks/REPORT.md`, and `docs/BENCHMARKING.md`.

Verification:

```bash
python benchmarks/run_benchmarks.py --model models/detector.onnx --source assets/sample/demo.mp4 --frames 200 --warmup 5
```

Acceptance: same host/model/source/frames/thresholds/warm-up/no-display settings are used; report includes environment details, count, inference and total latency statistics, FPS, caveats, and only actually measured values.

## Phase 8 - Documentation and Portfolio Polish

Goal: make the stable core reproducible and credible to reviewers.

Files: final `README.md`, architecture diagram, all `docs/` pages, screenshot, demo workflow, CI hardening, and third-party notices.

Verification:

```bash
scripts/smoke_test.sh
git status --short
git ls-files | rg '(\.onnx$|outputs/|\.venv/|third_party/|build/)'
```

Acceptance: fresh-clone instructions are complete; real commands/results replace placeholders; screenshot comes from this project; model/assets are attributed; no generated or local-only artifacts are tracked; all Definition of Done items are checked against evidence.

## Phase 9 - Optional Extensions

Start only after Phase 8 and the core Definition of Done are complete. Each extension requires its own plan and benchmark: CUDA, OpenVINO/TensorRT, asynchronous capture, batching, a second detector adapter, Docker, expanded CI matrix, or profiling.

## Risk Register and Open Decisions

1. **Python version:** Ubuntu currently exposes Python 3.10.12, below the required 3.11. Phase 0 Python verification needs a user-approved Python 3.11+ installation or an already-installed alternate interpreter.
2. **ONNX Runtime compatibility:** the 1.24.4 Python wheel installed successfully under Python 3.12, but the matching official C++ archive still needs Ubuntu 22.04/glibc verification before the exact shared version is locked.
3. **Model artifact licensing:** the YOLOX source repository is Apache-2.0, but the exact ONNX asset and COCO labels still need provenance and redistribution notes before use.
4. **Model output contract:** confirm whether the selected ONNX export contains decoded grids and record the exact output layout before implementing postprocessing.
5. **OpenCV split:** C++ OpenCV 4.5.4 is installed, but Python OpenCV is absent; Python should use a constrained package dependency in its isolated environment.
6. **C++ tests:** choose GoogleTest or Catch2 only after confirming the least fragile reproducible acquisition path; avoid hidden network downloads during normal CMake configure.
7. **Repository license:** owner selection is required before adding a license for original code.
8. **Demo assets/codecs:** use generated or clearly licensed small media; verify a codec available on the target Ubuntu host before promising saved video support.
9. **GUI/headless behavior:** automated tests and benchmarks default to `--no-display`; interactive display is verified separately where a desktop session exists.

## Phase Boundary Report Template

At the end of every phase, record:

- Scope completed and files changed.
- Exact verification commands run.
- Passed, failed, skipped, or blocked results with reasons.
- Dependency/version decisions made.
- Remaining limitations and the next phase, which must not start implicitly.
