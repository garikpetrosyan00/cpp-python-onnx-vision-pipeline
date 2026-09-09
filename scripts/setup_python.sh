#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python 3.11+ was not found. Set PYTHON_BIN to a compatible interpreter." >&2
    exit 1
fi

"${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
    echo "${python_bin} is older than Python 3.11." >&2
    exit 1
}

"${python_bin}" -m venv "${project_root}/python/.venv"
"${project_root}/python/.venv/bin/python" -m pip install --upgrade pip
"${project_root}/python/.venv/bin/python" -m pip install -e "${project_root}/python[dev]"

echo "Python environment ready at ${project_root}/python/.venv"
