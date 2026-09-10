"""Offline schema checks for the comparative benchmark runner."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_benchmarks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document(implementation: str) -> dict[str, object]:
    timings = {
        name: {
            "count": 2,
            "mean_ms": 1.0,
            "median_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.5,
            "p99_ms": 1.9,
            "min_ms": 0.5,
            "max_ms": 2.0,
        }
        for name in ("capture", "preprocess", "inference", "postprocess", "render", "total")
    }
    return {
        "schema_version": "1.0",
        "implementation": implementation,
        "mode": "detector",
        "source_kind": "video",
        "requested_warmup_frames": 5,
        "completed_warmup_frames": 5,
        "requested_measured_frames": 2,
        "completed_measured_frames": 2,
        "confidence": 0.25,
        "iou": 0.45,
        "no_display": True,
        "model_path": "models/detector.onnx",
        "model_sha256": "a" * 64,
        "provider": "CPUExecutionProvider",
        "versions": {"runtime": "1"},
        "approximate_process_rss": {"bytes": 1, "megabytes": 0.000001, "meaning": "snapshot"},
        "timing_boundaries": {"unit": "milliseconds", **{name: name for name in timings}},
        "percentile_method": "linear interpolation at sorted index (n - 1) * p / 100",
        "timings": timings,
        "effective_fps": 10.0,
    }


def test_benchmark_documents_validate_and_require_compatible_conditions() -> None:
    runner = _module()
    python = runner.validate_document(_document("python"), "python")
    cpp = runner.validate_document(_document("cpp"), "cpp")
    runner.compare_compatible(python, cpp)
    cpp["confidence"] = 0.3
    with pytest.raises(runner.BenchmarkValidationError, match="confidence"):
        runner.compare_compatible(python, cpp)


def test_benchmark_document_rejects_missing_timing_schema() -> None:
    runner = _module()
    document = _document("python")
    document["timings"].pop("total")  # type: ignore[index]
    with pytest.raises(runner.BenchmarkValidationError, match="timing names"):
        runner.validate_document(document, "python")
