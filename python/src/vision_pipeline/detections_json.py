"""Stable Phase 6 canonical detection export."""

import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from vision_pipeline.model_contract import MODEL_SHA256
from vision_pipeline.postprocess import Detection

SCHEMA_VERSION = "vision-pipeline-detections/v1"


class DetectionJsonError(ValueError):
    """Canonical detection document validation or publication failure."""


def document(
    detections: Sequence[Detection], image_size: tuple[int, int], confidence: float, iou: float
) -> dict[str, object]:
    height, width = image_size
    records: list[dict[str, object]] = []
    for detection in detections:
        values = (detection.confidence, detection.x1, detection.y1, detection.x2, detection.y2)
        if not all(math.isfinite(value) for value in values):
            raise DetectionJsonError("Canonical detections must contain finite numeric values.")
        records.append(
            {
                "class_id": detection.class_id,
                "label": detection.label,
                "confidence": float(detection.confidence),
                "x1": float(detection.x1),
                "y1": float(detection.y1),
                "x2": float(detection.x2),
                "y2": float(detection.y2),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": "python",
        "model_contract": {"name": "audited-yolox-nano", "sha256": MODEL_SHA256},
        "image": {"height": height, "width": width},
        "thresholds": {"confidence": confidence, "iou": iou},
        "detections": records,
    }


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    except (OSError, ValueError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DetectionJsonError(f"Cannot publish detections JSON {path}: {exc}") from exc
