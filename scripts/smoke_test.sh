#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"
build_dir="${BUILD_DIR:-${project_root}/build}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python 3.11+ was not found. Set PYTHON_BIN to a compatible interpreter." >&2
    exit 1
fi

"${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'

PYTHONPATH="${project_root}/python/src" "${python_bin}" -m vision_pipeline --help >/dev/null
PYTHONPATH="${project_root}/python/src" "${python_bin}" -m pytest "${project_root}/python/tests"

cmake -S "${project_root}/cpp" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j
ctest --test-dir "${build_dir}" --output-on-failure

echo "Phase 0 smoke checks passed."
