#!/usr/bin/env python3
"""Run compatible Python and C++ CPU benchmarks and validate their result documents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

SCHEMA_VERSION = "1.0"
TIMING_NAMES = ("capture", "preprocess", "inference", "postprocess", "render", "total")
SUMMARY_KEYS = {
    "count",
    "mean_ms",
    "median_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
}


class BenchmarkValidationError(ValueError):
    """A result document cannot be compared honestly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkValidationError(message)


def _finite_number(value: object, name: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{name} must be finite number.",
    )


def validate_document(document: object, implementation: str) -> dict[str, Any]:
    _require(isinstance(document, dict), "Benchmark JSON must be an object.")
    required = {
        "schema_version",
        "implementation",
        "mode",
        "source_kind",
        "requested_warmup_frames",
        "completed_warmup_frames",
        "requested_measured_frames",
        "completed_measured_frames",
        "confidence",
        "iou",
        "no_display",
        "model_path",
        "model_sha256",
        "provider",
        "versions",
        "approximate_process_rss",
        "timing_boundaries",
        "percentile_method",
        "timings",
        "effective_fps",
    }
    _require(
        set(document) == required,
        "Benchmark JSON has unexpected or missing top-level fields.",
    )
    _require(
        document["schema_version"] == SCHEMA_VERSION,
        "Unsupported benchmark schema version.",
    )
    _require(
        document["implementation"] == implementation,
        f"Expected implementation {implementation!r}.",
    )
    _require(document["mode"] == "detector", "Comparative runs require detector mode.")
    _require(
        document["source_kind"] in {"video", "camera", "image"}, "Invalid source kind."
    )
    for name in (
        "requested_warmup_frames",
        "completed_warmup_frames",
        "completed_measured_frames",
    ):
        _require(
            type(document[name]) is int and document[name] >= 0,
            f"{name} must be a non-negative integer.",
        )
    _require(
        document["requested_measured_frames"] is None
        or (
            type(document["requested_measured_frames"]) is int
            and document["requested_measured_frames"] > 0
        ),
        "requested_measured_frames must be null or a positive integer.",
    )
    _finite_number(document["confidence"], "confidence")
    _finite_number(document["iou"], "iou")
    _require(
        type(document["no_display"]) is bool and document["no_display"],
        "Benchmark must use --no-display.",
    )
    _require(
        isinstance(document["model_path"], str) and document["model_path"],
        "Model path is required.",
    )
    _require(
        isinstance(document["model_sha256"], str)
        and len(document["model_sha256"]) == 64,
        "Model SHA-256 is required.",
    )
    _require(
        document["provider"] == "CPUExecutionProvider",
        "Benchmark must use CPUExecutionProvider.",
    )
    _require(
        isinstance(document["versions"], dict) and document["versions"],
        "Versions object is required.",
    )
    rss = document["approximate_process_rss"]
    _require(
        isinstance(rss, dict) and set(rss) == {"bytes", "megabytes", "meaning"},
        "Invalid RSS object.",
    )
    _finite_number(rss["bytes"], "RSS bytes", allow_none=True)
    _finite_number(rss["megabytes"], "RSS megabytes", allow_none=True)
    _require(isinstance(rss["meaning"], str), "RSS meaning must be text.")
    _require(
        isinstance(document["timing_boundaries"], dict),
        "Timing boundaries must be an object.",
    )
    _require(
        set(document["timing_boundaries"]) == {"unit", *TIMING_NAMES},
        "Invalid timing boundary keys.",
    )
    _require(
        document["timing_boundaries"]["unit"] == "milliseconds",
        "Timing unit must be milliseconds.",
    )
    _require(
        all(
            isinstance(document["timing_boundaries"][name], str)
            for name in TIMING_NAMES
        ),
        "Timing boundaries must be text.",
    )
    _require(
        isinstance(document["percentile_method"], str),
        "Percentile method must be text.",
    )
    timings = document["timings"]
    _require(
        isinstance(timings, dict) and set(timings) == set(TIMING_NAMES),
        "Invalid timing names.",
    )
    for timing_name, summary in timings.items():
        _require(
            isinstance(summary, dict) and set(summary) == SUMMARY_KEYS,
            f"Invalid {timing_name} summary.",
        )
        _require(
            type(summary["count"]) is int and summary["count"] >= 0,
            f"Invalid {timing_name} count.",
        )
        for key in SUMMARY_KEYS - {"count"}:
            _finite_number(summary[key], f"{timing_name}.{key}", allow_none=True)
    _finite_number(document["effective_fps"], "effective_fps", allow_none=True)
    return document


def validate_csv(path: Path, implementation: str) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise BenchmarkValidationError(f"Cannot read CSV {path}: {exc}") from exc
    _require(len(rows) == 1, f"Benchmark CSV {path} must contain exactly one data row.")
    _require(
        rows[0].get("Implementation") == implementation,
        f"Benchmark CSV {path} implementation mismatch.",
    )


def compare_compatible(python: dict[str, Any], cpp: dict[str, Any]) -> None:
    for name in (
        "schema_version",
        "mode",
        "source_kind",
        "model_sha256",
        "provider",
        "confidence",
        "iou",
        "requested_warmup_frames",
        "completed_warmup_frames",
        "requested_measured_frames",
        "completed_measured_frames",
    ):
        _require(
            python[name] == cpp[name],
            f"Incompatible benchmark field {name}: Python={python[name]!r}, C++={cpp[name]!r}.",
        )


def source_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        _require(capture.isOpened(), f"Cannot open benchmark source {path}.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    _require(
        width > 0 and height > 0 and fps > 0 and frames > 0,
        "Benchmark source has invalid dimensions, FPS, or frame count.",
    )
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def command_line(
    args: argparse.Namespace, executable: str, output: Path, *, python_module: bool
) -> list[str]:
    command = [executable]
    if python_module:
        command.extend(("-m", "vision_pipeline"))
    return command + [
        "--model",
        str(args.model),
        "--labels",
        str(args.labels),
        "--source",
        str(args.source),
        "--no-display",
        "--benchmark",
        "--warmup",
        str(args.warmup),
        "--max-frames",
        str(args.frames),
        "--benchmark-output",
        str(output),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()
    try:
        _require(
            args.frames > 0 and args.warmup >= 0,
            "--frames must be positive and --warmup non-negative.",
        )
        _finite_number(args.confidence, "--confidence")
        _finite_number(args.iou, "--iou")
        _require(
            0 <= args.confidence <= 1 and 0 <= args.iou <= 1,
            "Thresholds must be in [0, 1].",
        )
        for path, name in (
            (args.python, "Python executable"),
            (args.cpp, "C++ executable"),
            (args.model, "model"),
            (args.labels, "labels"),
            (args.source, "source"),
        ):
            _require(path.is_file(), f"Missing {name}: {path}")
        metadata = source_metadata(args.source)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        python_output = args.results_dir / "python-benchmark.json"
        cpp_output = args.results_dir / "cpp-benchmark.json"
        subprocess.run(
            command_line(args, str(args.python), python_output, python_module=True),
            check=True,
        )
        subprocess.run(
            command_line(args, str(args.cpp), cpp_output, python_module=False),
            check=True,
        )
        python_document = validate_document(
            json.loads(python_output.read_text(encoding="utf-8")), "python"
        )
        cpp_document = validate_document(
            json.loads(cpp_output.read_text(encoding="utf-8")), "cpp"
        )
        validate_csv(python_output.with_suffix(".csv"), "python")
        validate_csv(cpp_output.with_suffix(".csv"), "cpp")
        compare_compatible(python_document, cpp_document)
        comparison = {
            "source": metadata,
            "python": python_document,
            "cpp": cpp_document,
        }
        (args.results_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Validated compatible benchmark results in {args.results_dir}")
        return 0
    except (
        BenchmarkValidationError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"benchmark comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
