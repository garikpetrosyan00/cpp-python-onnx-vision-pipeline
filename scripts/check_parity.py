"""Run both audited CLIs and compare Phase 6 canonical detection documents."""

import argparse
import json
import math
import subprocess
from pathlib import Path

FIELDS = ("class_id", "label", "confidence", "x1", "y1", "x2", "y2")
FLOAT_TOLERANCE = 1e-5


def load_document(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid detections JSON {path}: {error}") from error
    required = {
        "schema_version",
        "implementation",
        "model_contract",
        "image",
        "thresholds",
        "detections",
    }
    if (
        not isinstance(data, dict)
        or set(data) != required
        or data["schema_version"] != "vision-pipeline-detections/v1"
    ):
        raise ValueError(f"Invalid canonical schema in {path}.")
    if not isinstance(data["implementation"], str):
        raise TypeError(f"Invalid implementation in {path}.")
    for field, keys in (
        ("model_contract", {"name", "sha256"}),
        ("image", {"height", "width"}),
        ("thresholds", {"confidence", "iou"}),
    ):
        if not isinstance(data[field], dict) or set(data[field]) != keys:
            raise TypeError(f"Invalid {field} metadata in {path}.")
    if not all(
        isinstance(data["model_contract"][key], str) for key in data["model_contract"]
    ):
        raise TypeError(f"Invalid model_contract types in {path}.")
    if not all(
        type(data["image"][key]) is int and data["image"][key] > 0
        for key in data["image"]
    ):
        raise TypeError(f"Invalid image dimensions in {path}.")
    if not all(
        type(data["thresholds"][key]) in (int, float)
        and math.isfinite(data["thresholds"][key])
        and 0 <= data["thresholds"][key] <= 1
        for key in data["thresholds"]
    ):
        raise TypeError(f"Invalid threshold values in {path}.")
    if not isinstance(data["detections"], list):
        raise TypeError(f"Invalid detections list in {path}.")
    for index, detection in enumerate(data["detections"]):
        if not isinstance(detection, dict) or set(detection) != set(FIELDS):
            raise ValueError(f"Invalid detection {index} in {path}.")
        if (
            type(detection["class_id"]) is not int
            or detection["class_id"] < 0
            or not isinstance(detection["label"], str)
            or not detection["label"]
        ):
            raise TypeError(f"Invalid detection types at {index} in {path}.")
        for field in FIELDS[2:]:
            if type(detection[field]) not in (int, float) or not math.isfinite(
                detection[field]
            ):
                raise ValueError(f"Non-finite {field} at {index} in {path}.")
    return data


def compare(
    left: dict[str, object], right: dict[str, object]
) -> tuple[int, float, float]:
    if left["implementation"] != "python" or right["implementation"] != "cpp":
        raise ValueError(
            "Implementation identity mismatch: expected Python document then C++ document."
        )
    for section, field in (
        ("", "schema_version"),
        ("model_contract", "name"),
        ("model_contract", "sha256"),
        ("image", "height"),
        ("image", "width"),
        ("thresholds", "confidence"),
        ("thresholds", "iou"),
    ):
        python_value = left[field] if not section else left[section][field]
        cpp_value = right[field] if not section else right[section][field]
        if python_value != cpp_value:
            qualified = f"{section}.{field}" if section else field
            raise ValueError(
                f"Metadata mismatch field={qualified}: Python={python_value!r}, C++={cpp_value!r}."
            )
    a, b = left["detections"], right["detections"]
    if len(a) != len(b):
        raise ValueError(f"Detection count differs: Python={len(a)}, C++={len(b)}.")
    maximum_confidence = 0.0
    maximum_coordinate = 0.0
    for index, (python_detection, cpp_detection) in enumerate(zip(a, b)):
        for field in ("class_id", "label"):
            if python_detection[field] != cpp_detection[field]:
                raise ValueError(
                    f"Mismatch index={index} field={field}: Python={python_detection[field]!r}, C++={cpp_detection[field]!r}."
                )
        for field in FIELDS[2:]:
            delta = abs(python_detection[field] - cpp_detection[field])
            if field == "confidence":
                maximum_confidence = max(maximum_confidence, delta)
            else:
                maximum_coordinate = max(maximum_coordinate, delta)
            if delta > FLOAT_TOLERANCE:
                raise ValueError(
                    f"Mismatch index={index} field={field}: Python={python_detection[field]}, C++={cpp_detection[field]}, absolute_difference={delta}, tolerance={FLOAT_TOLERANCE}."
                )
    return len(a), maximum_confidence, maximum_coordinate


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("python", "cpp", "model", "labels", "image", "work-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    paths = {
        name.replace("-", "_"): Path(getattr(args, name.replace("-", "_")))
        for name in ("cpp", "model", "labels", "image")
    }
    if not Path(args.python).is_file() or any(
        not path.is_file() for path in paths.values()
    ):
        raise SystemExit(
            "Parity prerequisites missing: provide Python, C++ executable, audited model, labels, and image."
        )
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    python_json, cpp_json = work / "python.json", work / "cpp.json"
    common = [
        "--model",
        args.model,
        "--labels",
        args.labels,
        "--source",
        args.image,
        "--no-display",
        "--max-frames",
        "1",
        "--confidence",
        "0.01",
        "--iou",
        "0.45",
    ]
    for command in (
        [
            args.python,
            "-m",
            "vision_pipeline",
            *common,
            "--detections-json",
            str(python_json),
        ],
        [args.cpp, *common, "--detections-json", str(cpp_json)],
    ):
        subprocess.run(command, check=True)
    count, maximum_confidence, maximum_coordinate = compare(
        load_document(python_json), load_document(cpp_json)
    )
    if count == 0:
        raise ValueError(
            "Parity fixture produced zero detections; coordinate parity was not exercised."
        )
    print(
        f"Parity passed: detections={count}, maximum_confidence_delta={maximum_confidence}, "
        f"maximum_coordinate_delta={maximum_coordinate}, tolerance={FLOAT_TOLERANCE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
