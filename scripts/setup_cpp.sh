#!/usr/bin/env bash
set -euo pipefail

readonly ORT_VERSION="1.24.4"
readonly ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
readonly ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
readonly ORT_BYTES=8155822
readonly ORT_SHA256="3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747"
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DESTINATION="${ROOT_DIR}/third_party/onnxruntime"

fail() { echo "setup_cpp.sh: error: $*" >&2; exit 1; }
verify_tree() { [[ -f "${1}/include/onnxruntime_cxx_api.h" && -f "${1}/lib/libonnxruntime.so" ]]; }
for required_command in cmake c++ pkg-config curl tar sha256sum; do
    command -v "${required_command}" >/dev/null 2>&1 || fail "missing required command: ${required_command}"
done
pkg-config --exists opencv4 || fail "OpenCV 4 development files were not found through pkg-config."
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "this setup supports Linux x86-64 only."
if [[ -e "${DESTINATION}" ]]; then
    verify_tree "${DESTINATION}" || fail "${DESTINATION} exists but is incomplete. Remove it and rerun this script."
    echo "ONNX Runtime ${ORT_VERSION} is already verified at ${DESTINATION}"
    exit 0
fi
mkdir -p "${ROOT_DIR}/third_party"
temporary_archive="$(mktemp "${ROOT_DIR}/third_party/.${ORT_ARCHIVE}.XXXXXX")"
temporary_directory="$(mktemp -d "${ROOT_DIR}/third_party/.onnxruntime.extract.XXXXXX")"
cleanup() { rm -f "${temporary_archive}"; rm -rf "${temporary_directory}"; }
trap cleanup EXIT
echo "Downloading ONNX Runtime ${ORT_VERSION} CPU archive from ${ORT_URL}"
curl --fail --location --retry 3 --retry-delay 1 --output "${temporary_archive}" "${ORT_URL}" || fail "download failed. Check network access and retry."
observed_bytes="$(wc -c < "${temporary_archive}")"
observed_sha="$(sha256sum "${temporary_archive}" | awk '{print $1}')"
[[ "${observed_bytes}" == "${ORT_BYTES}" && "${observed_sha}" == "${ORT_SHA256}" ]] || fail "archive verification failed: observed ${observed_bytes} bytes, SHA-256 ${observed_sha}; expected ${ORT_BYTES} bytes, SHA-256 ${ORT_SHA256}."
tar -xzf "${temporary_archive}" -C "${temporary_directory}" || fail "cannot extract verified archive."
extracted="${temporary_directory}/onnxruntime-linux-x64-${ORT_VERSION}"
verify_tree "${extracted}" || fail "verified archive lacks expected headers or lib/libonnxruntime.so."
mv "${extracted}" "${DESTINATION}" || fail "cannot publish ONNX Runtime to ${DESTINATION}."
echo "Installed ONNX Runtime ${ORT_VERSION} at ${DESTINATION}"
echo "Configure with: -DONNXRUNTIME_ROOT=${DESTINATION}"
