# Troubleshooting

## Setup

- Python requires 3.11 or newer. Select a compatible interpreter with `PYTHON_BIN=/path/to/python scripts/setup_python.sh`.
- CMake needs OpenCV 4 development files discoverable through `find_package(OpenCV)`; on Ubuntu install `libopencv-dev`.
- Run `scripts/setup_cpp.sh` on Linux x86-64 to fetch the pinned ONNX Runtime 1.24.4 CPU archive into ignored `third_party/onnxruntime/`. It verifies 8,155,822 bytes and SHA-256 `3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747` before publishing it.
- Configure C++ with `-DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime"`. The build embeds that directory as a runtime search path, so no global linker configuration is required.
- If the model command fails, run `python/.venv/bin/python scripts/download_or_export_model.py --verify-only`; download it with the same script when it is absent. The model is intentionally ignored by Git.

## Runtime

- Both detector paths are CPU-only and reject models that do not match the audited YOLOX-Nano metadata and SHA-256.
- Use `--no-display` on servers or systems without a working OpenCV GUI backend. Interactive Q/ESC behavior needs a desktop session.
- Camera availability depends on attached hardware and permissions. Automated tests mock camera/GUI behavior.
- Video input/output support depends on the installed OpenCV backend and codecs. Prefer `.avi`/MJPG when an encoder is unavailable; output is verified before publication.
- Parity and benchmark commands require the local model, labels, C++ executable, and ONNX Runtime setup; they fail rather than silently skip if a prerequisite is missing.
