# Troubleshooting

## Current Phase 0 Checks

- Python requires version 3.11 or newer. Select it with `PYTHON_BIN=/path/to/python scripts/smoke_test.sh`.
- CMake requires OpenCV 4 development files discoverable by `find_package(OpenCV)`.
- ONNX Runtime C++ is intentionally not required until its pinned local setup is implemented.
- Automated checks are headless; later interactive runs will document `--no-display` for systems without a GUI session.
