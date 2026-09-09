"""The audited artifact identity; stdlib only so tooling/help stay lightweight."""

import hashlib
from pathlib import Path

MODEL_URL = "https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/assets/42724905"
MODEL_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
MODEL_BYTES = 3_659_407
INPUT_SHAPE = (1, 3, 416, 416)
OUTPUT_SHAPE = (1, 3549, 85)
CLASS_COUNT = 80
PROVIDER = "CPUExecutionProvider"
EXPECTED_CONTRACT = (
    "audited YOLOX-Nano, one float32 input [1,3,416,416], "
    "one float32 raw output [1,3549,85], external stride-8/16/32 decoding"
)


class DetectorError(RuntimeError):
    """Model acquisition, inference, or tensor-contract failure."""


def model_context(
    path: Path, detail: str, observed: str = "unavailable (session not loaded)"
) -> str:
    return (
        f"Model {path}; provider {PROVIDER}; observed {observed}; "
        f"expected {EXPECTED_CONTRACT}. {detail}"
    )


def verify_model_file(path: Path) -> None:
    """Reject missing, unreadable, truncated, modified, or different model bytes."""
    try:
        if not path.is_file():
            raise DetectorError(f"Model is missing or not a regular file: {path}.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = path.stat().st_size
    except OSError as exc:
        raise DetectorError(
            f"Cannot read model {path}: {exc}. Check path and permissions."
        ) from exc
    if size != MODEL_BYTES or digest != MODEL_SHA256:
        raise DetectorError(
            f"Model checksum/size mismatch at {path}: observed SHA-256 {digest}, {size} bytes; "
            f"expected SHA-256 {MODEL_SHA256}, {MODEL_BYTES} bytes. "
            "Run scripts/download_or_export_model.py to obtain the audited artifact."
        )
