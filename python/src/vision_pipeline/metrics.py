"""Deterministic Phase 3 timing collection and benchmark result publication."""

import csv
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from vision_pipeline.config import PipelineConfig
from vision_pipeline.model_contract import MODEL_SHA256, PROVIDER

SCHEMA_VERSION = "1.0"
TIMING_NAMES = ("capture", "preprocess", "inference", "postprocess", "render", "total")


class BenchmarkError(RuntimeError):
    """An actionable benchmark collection or publication failure."""


@dataclass(frozen=True)
class FrameTiming:
    """Raw non-negative nanoseconds. Total is exactly the six processing boundaries' sum."""

    capture_ns: int
    preprocess_ns: int
    inference_ns: int
    postprocess_ns: int
    render_ns: int
    total_ns: int

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Frame timings must be non-negative integer nanoseconds.")
        if self.total_ns != sum(values[:-1]):
            raise ValueError("total_ns must equal capture through render nanoseconds exactly.")


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float | None
    median_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    min_ms: float | None
    max_ms: float | None


def percentile(samples: list[int] | tuple[int, ...], percent: int) -> float | None:
    """Linear interpolation between sorted ranks at index ``(n - 1) * percent / 100``."""
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be in [0, 100].")
    if not samples:
        return None
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(samples_ns: list[int] | tuple[int, ...]) -> TimingSummary:
    if any(type(value) is not int or value < 0 for value in samples_ns):
        raise ValueError("Timing samples must be non-negative integer nanoseconds.")
    if not samples_ns:
        return TimingSummary(0, None, None, None, None, None, None, None)
    values = sorted(samples_ns)
    to_ms = lambda value: value / 1_000_000  # noqa: E731
    return TimingSummary(
        count=len(values),
        mean_ms=to_ms(sum(values) / len(values)),
        median_ms=to_ms(percentile(values, 50)),
        p50_ms=to_ms(percentile(values, 50)),
        p95_ms=to_ms(percentile(values, 95)),
        p99_ms=to_ms(percentile(values, 99)),
        min_ms=to_ms(values[0]),
        max_ms=to_ms(values[-1]),
    )


def effective_fps(samples_ns: list[int] | tuple[int, ...]) -> float | None:
    if any(type(value) is not int or value < 0 for value in samples_ns):
        raise ValueError("Timing samples must be non-negative integer nanoseconds.")
    total = sum(samples_ns)
    return len(samples_ns) * 1_000_000_000 / total if total else None


class MetricsCollector:
    def __init__(self) -> None:
        self._frames: list[FrameTiming] = []

    def add(self, timing: FrameTiming) -> None:
        self._frames.append(timing)

    @property
    def frames(self) -> tuple[FrameTiming, ...]:
        return tuple(self._frames)

    def summaries(self) -> dict[str, TimingSummary]:
        return {
            name: timing_summary([getattr(frame, f"{name}_ns") for frame in self._frames])
            for name in TIMING_NAMES
        }

    def fps(self) -> float | None:
        return effective_fps([frame.total_ns for frame in self._frames])


def approximate_rss_bytes() -> int | None:
    """Current process RSS, an approximate process-wide snapshot rather than a peak measurement."""
    try:
        return psutil.Process().memory_info().rss
    except (psutil.Error, OSError):
        return None


def benchmark_document(
    config: PipelineConfig,
    collector: MetricsCollector,
    warmup_completed: int,
    requested_warmup: int,
    requested_frames: int | None,
) -> dict[str, Any]:
    summaries = collector.summaries()
    rss = approximate_rss_bytes()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "implementation": "python",
        "mode": "detector" if config.model is not None else "passthrough",
        "source_kind": config.source.kind.value,
        "requested_warmup_frames": requested_warmup,
        "completed_warmup_frames": warmup_completed,
        "requested_measured_frames": requested_frames,
        "completed_measured_frames": len(collector.frames),
        "confidence": config.confidence,
        "iou": config.iou,
        "no_display": config.no_display,
        "model_path": str(config.model) if config.model is not None else None,
        "model_sha256": MODEL_SHA256 if config.model is not None else None,
        "provider": PROVIDER if config.model is not None else None,
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": _onnxruntime_version() if config.model is not None else None,
        },
        "approximate_process_rss": {
            "bytes": rss,
            "megabytes": rss / 1_000_000 if rss is not None else None,
            "meaning": "current process RSS snapshot after measured frames; not peak memory",
        },
        "timing_boundaries": {
            "unit": "milliseconds",
            "capture": "InputSource next(frame) only",
            "preprocess": "BGR frame to model tensor; zero in passthrough",
            "inference": "InferenceEngine.run only; zero in passthrough",
            "postprocess": (
                "decode, score filter, class-aware NMS, restore, clamp; zero in passthrough"
            ),
            "render": "annotation, output write, and GUI draw; excludes cv2.waitKey delay",
            "total": (
                "sum of capture, preprocess, inference, postprocess, render; excludes GUI wait"
            ),
        },
        "percentile_method": "linear interpolation at sorted index (n - 1) * p / 100",
        "timings": {name: asdict(summary) for name, summary in summaries.items()},
        "effective_fps": collector.fps(),
    }
    return document


def _onnxruntime_version() -> str:
    import onnxruntime

    return onnxruntime.__version__


CSV_COLUMNS = (
    "Implementation",
    "Mode",
    "Frames",
    "WarmupFrames",
    "CaptureMeanMs",
    "PreprocessMeanMs",
    "InferenceMeanMs",
    "InferenceP50Ms",
    "InferenceP95Ms",
    "InferenceP99Ms",
    "PostprocessMeanMs",
    "RenderMeanMs",
    "TotalMeanMs",
    "TotalP50Ms",
    "TotalP95Ms",
    "TotalP99Ms",
    "FPS",
    "RSS_MB",
)


def csv_row(document: dict[str, Any]) -> dict[str, Any]:
    timings = document["timings"]
    return {
        "Implementation": document["implementation"],
        "Mode": document["mode"],
        "Frames": document["completed_measured_frames"],
        "WarmupFrames": document["completed_warmup_frames"],
        "CaptureMeanMs": timings["capture"]["mean_ms"],
        "PreprocessMeanMs": timings["preprocess"]["mean_ms"],
        "InferenceMeanMs": timings["inference"]["mean_ms"],
        "InferenceP50Ms": timings["inference"]["p50_ms"],
        "InferenceP95Ms": timings["inference"]["p95_ms"],
        "InferenceP99Ms": timings["inference"]["p99_ms"],
        "PostprocessMeanMs": timings["postprocess"]["mean_ms"],
        "RenderMeanMs": timings["render"]["mean_ms"],
        "TotalMeanMs": timings["total"]["mean_ms"],
        "TotalP50Ms": timings["total"]["p50_ms"],
        "TotalP95Ms": timings["total"]["p95_ms"],
        "TotalP99Ms": timings["total"]["p99_ms"],
        "FPS": document["effective_fps"],
        "RSS_MB": document["approximate_process_rss"]["megabytes"],
    }


def write_benchmark_results(destination: Path, document: dict[str, Any]) -> tuple[Path, Path]:
    """Stage both files, then replace both destinations while restoring old files on failure."""
    if destination.suffix.lower() != ".json":
        raise BenchmarkError(
            "--benchmark-output must end in .json so the sibling CSV is unambiguous."
        )
    csv_destination = destination.with_suffix(".csv")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _validate_destinations((destination, csv_destination))
        json_temp = _stage(destination, json.dumps(document, indent=2, sort_keys=True) + "\n")
        csv_temp = _stage_csv(csv_destination, csv_row(document))
        _publish_pair(((json_temp, destination), (csv_temp, csv_destination)))
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkError(
            f"Cannot publish benchmark results at {destination}: {exc}. "
            "Check path, permissions, free space, and JSON-compatible result data."
        ) from exc
    return destination, csv_destination


def _validate_destinations(paths: tuple[Path, Path]) -> None:
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("JSON and CSV benchmark destinations must be distinct.")
    for path in paths:
        if path.exists() and not path.is_file():
            raise ValueError(f"Destination is not a regular file: {path}")


def _stage(destination: Path, contents: str) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=f".{destination.stem}-",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _stage_csv(destination: Path, row: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.stem}-",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _publish_pair(pairs: tuple[tuple[Path, Path], tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, destination in pairs:
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup")
                if backup.exists():
                    raise OSError(f"Temporary backup already exists: {backup}")
                destination.replace(backup)
                backups.append((backup, destination))
        for temporary, destination in pairs:
            temporary.replace(destination)
            published.append(destination)
    except OSError:
        for destination in published:
            destination.unlink(missing_ok=True)
        for backup, destination in backups:
            if backup.exists():
                backup.replace(destination)
        raise
    finally:
        for temporary, _ in pairs:
            temporary.unlink(missing_ok=True)
        for backup, _ in backups:
            backup.unlink(missing_ok=True)
