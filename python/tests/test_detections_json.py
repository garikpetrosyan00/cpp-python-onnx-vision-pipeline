import json

import pytest

from vision_pipeline.detections_json import document, write_atomic
from vision_pipeline.postprocess import Detection


def test_document_is_canonical_and_ordered() -> None:
    detections = [Detection(2, "car", 0.5, 1, 2, 3, 4), Detection(1, "bike", 0.4, 5, 6, 7, 8)]
    payload = document(detections, (416, 416), 0.01, 0.45)
    assert payload["schema_version"] == "vision-pipeline-detections/v1"
    assert list(payload["detections"][0]) == [
        "class_id",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    assert [item["class_id"] for item in payload["detections"]] == [2, 1]


def test_write_atomic_preserves_existing_destination_on_serialization_failure(tmp_path) -> None:
    destination = tmp_path / "detections.json"
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        write_atomic(destination, {"bad": float("nan")})
    assert destination.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_write_atomic_writes_readable_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "detections.json"
    write_atomic(destination, document([], (416, 416), 0.01, 0.45))
    assert json.loads(destination.read_text(encoding="utf-8"))["detections"] == []
