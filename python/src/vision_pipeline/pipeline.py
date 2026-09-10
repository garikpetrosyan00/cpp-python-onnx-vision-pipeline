"""Media passthrough or acquire → preprocess → infer → postprocess → render."""

from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns

from vision_pipeline.config import PipelineConfig
from vision_pipeline.detections_json import document as detections_document
from vision_pipeline.detections_json import write_atomic
from vision_pipeline.input_source import InputSource
from vision_pipeline.metrics import (
    BenchmarkError,
    FrameTiming,
    MetricsCollector,
    benchmark_document,
    write_benchmark_results,
)
from vision_pipeline.postprocess import postprocess
from vision_pipeline.preprocess import preprocess
from vision_pipeline.render import Renderer


def _default_benchmark_output() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "benchmarks" / "results" / "python-benchmark.json"


def run_pipeline(config: PipelineConfig) -> int:
    """Validate the detector before acquisition; own all resources until exit."""
    frames = 0
    canonical_detections = []
    canonical_size: tuple[int, int] | None = None
    with ExitStack() as resources:
        detector = None
        if config.model is not None:
            from vision_pipeline.inference import InferenceEngine

            detector = resources.enter_context(InferenceEngine(config.model))
        source = resources.enter_context(InputSource(config.source))
        renderer = resources.enter_context(Renderer(config, source.fps))
        collector = MetricsCollector()
        warmup_completed = 0
        while True:
            capture_start = perf_counter_ns()
            try:
                frame = next(source)
            except StopIteration:
                break
            capture_ns = perf_counter_ns() - capture_start
            detections = []
            preprocess_ns = inference_ns = postprocess_ns = 0
            if detector is not None:
                started = perf_counter_ns()
                prepared = preprocess(frame)
                preprocess_ns = perf_counter_ns() - started
                started = perf_counter_ns()
                output = detector.run(prepared.tensor)
                inference_ns = perf_counter_ns() - started
                started = perf_counter_ns()
                detections = postprocess(
                    output, prepared.metadata, config.label_names, config.confidence, config.iou
                )
                postprocess_ns = perf_counter_ns() - started
                if config.detections_json is not None:
                    canonical_detections = detections
                    canonical_size = frame.shape[:2]
            started = perf_counter_ns()
            renderer.render_processing(frame, detections)
            render_ns = perf_counter_ns() - started
            timing = FrameTiming(
                capture_ns,
                preprocess_ns,
                inference_ns,
                postprocess_ns,
                render_ns,
                capture_ns + preprocess_ns + inference_ns + postprocess_ns + render_ns,
            )
            is_warmup = config.benchmark and warmup_completed < config.warmup
            if is_warmup:
                warmup_completed += 1
            else:
                frames += 1
                if config.benchmark:
                    collector.add(timing)
            if not renderer.wait_for_exit():
                break
            if config.max_frames is not None and frames >= config.max_frames:
                break
        if config.benchmark:
            if not collector.frames:
                raise BenchmarkError(
                    "Benchmark reached end-of-file before one measured frame. "
                    "Use fewer --warmup frames or a longer source."
                )
            destination = config.benchmark_output or _default_benchmark_output()
            document = benchmark_document(
                config, collector, warmup_completed, config.warmup, config.max_frames
            )
            write_benchmark_results(destination, document)
        if config.detections_json is not None:
            assert canonical_size is not None
            write_atomic(
                config.detections_json,
                detections_document(
                    canonical_detections, canonical_size, config.confidence, config.iou
                ),
            )
    return frames
