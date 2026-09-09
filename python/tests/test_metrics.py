import csv
import json
from pathlib import Path

import pytest

from vision_pipeline.config import PipelineConfig, parse_source
from vision_pipeline.metrics import (
    BenchmarkError,
    FrameTiming,
    MetricsCollector,
    benchmark_document,
    effective_fps,
    percentile,
    timing_summary,
    write_benchmark_results,
)


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([], 0, None),
        ([7], 0, 7),
        ([7], 100, 7),
        ([10, 0], 50, 5),
        ([1, 3, 5, 7], 50, 4),
        ([1, 3, 5, 7], 95, 6.7),
        ([4, 4, 4], 99, 4),
        ([10**18, 1], 100, 10**18),
    ],
)
def test_percentile_linear_interpolation(values, percent, expected) -> None:
    assert percentile(values, percent) == expected


@pytest.mark.parametrize("percent", [-1, 101])
def test_invalid_percentile(percent: int) -> None:
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1], percent)


def test_timing_summary_and_fps_are_exact() -> None:
    summary = timing_summary([3_000_000, 1_000_000, 1_000_000, 5_000_000])
    assert summary.count == 4
    assert summary.mean_ms == 2.5
    assert summary.median_ms == summary.p50_ms == 2.0
    assert summary.p95_ms == 4.7
    assert summary.p99_ms == 4.94
    assert summary.min_ms == 1.0 and summary.max_ms == 5.0
    assert timing_summary([]).count == 0
    assert timing_summary([]).mean_ms is None
    assert effective_fps([1_000_000_000, 1_000_000_000]) == 1.0
    assert effective_fps([0, 0]) is None


@pytest.mark.parametrize("samples", [[-1], [1.0], [True]])
def test_invalid_samples_and_timings(samples) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        timing_summary(samples)
    with pytest.raises(ValueError, match="non-negative integer"):
        effective_fps(samples)
    with pytest.raises(ValueError):
        FrameTiming(1, 2, 3, 4, 5, 14)
    with pytest.raises(ValueError):
        FrameTiming(1, 2, 3, 4, -5, 5)


def _document() -> dict:
    config = PipelineConfig(parse_source("0"), no_display=True, benchmark=True, warmup=2)
    collector = MetricsCollector()
    collector.add(FrameTiming(1, 0, 0, 0, 2, 3))
    collector.add(FrameTiming(3, 0, 0, 0, 4, 7))
    return benchmark_document(config, collector, 2, 2, 2)


def test_document_and_csv_schema(tmp_path: Path) -> None:
    document = _document()
    assert document["mode"] == "passthrough"
    assert document["timings"]["preprocess"]["mean_ms"] == 0
    assert document["effective_fps"] == 2_000_000_000 / 10
    assert document["model_path"] is None and document["provider"] is None
    output = tmp_path / "deep" / "result.json"
    json_path, csv_path = write_benchmark_results(output, document)
    assert json.loads(json_path.read_text()) == document
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["Implementation"] == "python"
    assert rows[0]["Frames"] == "2"
    assert rows[0]["InferenceMeanMs"] == "0.0"
    assert not list(output.parent.glob("*.tmp"))


def test_atomic_publish_preserves_existing_pair_on_failure(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "result.json"
    csv_path = output.with_suffix(".csv")
    output.write_text('{"old": true}\n')
    csv_path.write_text("old,csv\n")
    original_replace = Path.replace

    def fail_csv(source: Path, destination: Path):
        if source.suffix == ".tmp" and destination == csv_path:
            raise PermissionError("denied")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_csv)
    with pytest.raises(BenchmarkError, match="Cannot publish"):
        write_benchmark_results(output, _document())
    assert output.read_text() == '{"old": true}\n'
    assert csv_path.read_text() == "old,csv\n"
    assert not list(tmp_path.glob(".*"))


def test_result_destination_errors(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="end in .json"):
        write_benchmark_results(tmp_path / "result.csv", _document())
    directory = tmp_path / "result.json"
    directory.mkdir()
    with pytest.raises(BenchmarkError, match="not a regular file"):
        write_benchmark_results(directory, _document())
