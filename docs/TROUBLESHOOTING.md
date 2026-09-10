# Troubleshooting

## C++ ONNX Runtime (Phase 5)

- Python requires version 3.11 or newer. Select it with `PYTHON_BIN=/path/to/python scripts/smoke_test.sh`.
- CMake requires OpenCV 4 development files discoverable by `find_package(OpenCV)`.
- Run `scripts/setup_cpp.sh` on Linux x86-64 to fetch the pinned ONNX Runtime 1.24.4 CPU archive into ignored `third_party/onnxruntime/`. It verifies 8,155,822 bytes and SHA-256 `3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747` before publishing it.
- Configure with `-DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime"`. CMake intentionally rejects missing or malformed roots instead of searching global headers/libraries. The build embeds that directory as a runtime search path; no global linker configuration is needed.
- C++ inference is CPU-only and accepts only the audited YOLOX-Nano model metadata. Use `--no-display` where no desktop session is available.
- Automated checks are headless; later interactive runs will document `--no-display` for systems without a GUI session.
