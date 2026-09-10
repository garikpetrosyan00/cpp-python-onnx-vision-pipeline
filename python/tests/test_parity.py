import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_parity.py"
SPEC = importlib.util.spec_from_file_location("check_parity", SCRIPT)
assert SPEC and SPEC.loader
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def canonical(implementation: str) -> dict[str, object]:
    return {
        "schema_version": "vision-pipeline-detections/v1",
        "implementation": implementation,
        "model_contract": {"name": "audited-yolox-nano", "sha256": "a" * 64},
        "image": {"height": 416, "width": 416},
        "thresholds": {"confidence": 0.01, "iou": 0.45},
        "detections": [
            {
                "class_id": 3,
                "label": "car",
                "confidence": 0.5,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
            }
        ],
    }


def test_valid_nonempty_comparison() -> None:
    assert parity.compare(canonical("python"), canonical("cpp")) == (1, 0.0, 0.0)


def test_cpp_implementation_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="Implementation identity"):
        parity.compare(canonical("python"), canonical("python"))


def test_tolerance_diagnostic_identifies_record_and_field() -> None:
    cpp = canonical("cpp")
    cpp["detections"][0]["x2"] += parity.FLOAT_TOLERANCE * 2
    with pytest.raises(ValueError, match="index=0 field=x2"):
        parity.compare(canonical("python"), cpp)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda value: value.update(implementation="cpp"), "Implementation"),
        (lambda value: value["model_contract"].update(name="other"), "model_contract.name"),
        (lambda value: value["model_contract"].update(sha256="b" * 64), "model_contract.sha256"),
        (lambda value: value["image"].update(width=415), "image.width"),
        (lambda value: value["thresholds"].update(iou=0.4), "thresholds.iou"),
        (lambda value: value["detections"].append(value["detections"][0]), "count"),
        (lambda value: value["detections"][0].update(class_id=4), "class_id"),
        (lambda value: value["detections"][0].update(label="bus"), "label"),
        (lambda value: value["detections"][0].update(x1=1.001), "x1"),
    ],
)
def test_comparison_failures(mutation, field: str) -> None:
    python = canonical("python")
    cpp = copy.deepcopy(canonical("cpp"))
    if field == "Implementation":
        mutation(python)
    else:
        mutation(cpp)
    with pytest.raises(ValueError, match=field):
        parity.compare(python, cpp)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(model_contract={"name": "x"}),
        lambda value: value["detections"][0].update(confidence=float("nan")),
    ],
)
def test_schema_validation_failures(tmp_path, mutation) -> None:
    payload = canonical("python")
    mutation(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        parity.load_document(path)
