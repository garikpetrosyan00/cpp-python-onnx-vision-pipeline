#!/usr/bin/env bash
set -euo pipefail

for required_command in cmake c++ pkg-config; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Missing required command: ${required_command}" >&2
        exit 1
    fi
done

if ! pkg-config --exists opencv4; then
    echo "OpenCV 4 development files were not found through pkg-config." >&2
    exit 1
fi

echo "CMake: $(cmake --version | head -n 1)"
echo "Compiler: $(c++ --version | head -n 1)"
echo "OpenCV: $(pkg-config --modversion opencv4)"

if [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
    echo "ONNXRUNTIME_ROOT is set to ${ONNXRUNTIME_ROOT}"
else
    echo "ONNX Runtime C++ setup is deferred until the pinned archive is validated."
fi
