# Python benchmarking (Phase 3)

Phase 3 adds Python-only, headless benchmark output. C++ benchmarking and cross-language comparison are Phase 7 work and have not started.

```bash
python/.venv/bin/python -m vision_pipeline \
  --model models/detector.onnx \
  --labels models/classes.txt \
  --source path/to/input.avi \
  --benchmark --no-display --warmup 5 --max-frames 200 \
  --benchmark-output benchmarks/results/python-run.json
```

`--benchmark` requires `--no-display`; the command exits with configuration status 2 otherwise. It also works without `--model` to measure the Phase 1 passthrough path. `--warmup` is a non-negative integer and defaults to 5. Warm-up frames follow the full capture, detector (if enabled), annotation/output, and headless-render path, but contribute no samples. In benchmark mode `--max-frames` limits measured frames only, excluding warm-up. Without `--benchmark`, `--max-frames` retains its Phase 1/2 meaning of all processed frames.

End-of-file before the first measured frame is an error. A clean run writes JSON to the explicit `--benchmark-output` or to `benchmarks/results/python-benchmark.json`, plus a sibling `.csv` file with the same base name. Both files are staged in the destination directory, then published together; failed collection or publication preserves existing results. Ctrl+C never publishes a benchmark result, even if it finalizes already-written annotated media.

All timing values begin as raw integer nanoseconds from `time.perf_counter_ns()`. `capture_ms` measures only `next(InputSource)`. `preprocess_ms` is BGR frame to tensor; `inference_ms` is only `InferenceEngine.run`; `postprocess_ms` covers decode/filter/NMS/restoration/clamping; `render_ms` covers annotation, output writing, and GUI drawing. The intentional `cv2.waitKey` delay is excluded. `total_ms` is exactly the sum of those processing boundaries, and effective FPS is `measured_frames * 1e9 / sum(total_ns)`. Passthrough reports exactly zero preprocess, inference, and postprocess timing; it does not invent unavailable detector measurements.

Each timing summary has count, mean, median/p50, p95, p99, minimum, and maximum in milliseconds. Percentiles use linear interpolation between sorted samples at index `(n - 1) * p / 100`; p50 is the median. An empty set has count zero and null summary/FPS values. The JSON records schema version, mode, source kind, requested/completed warm-up and measured counts, thresholds, display setting, model/provider identity when applicable, dependency versions, all timing definitions/summaries, and a current-process RSS snapshot. RSS is approximate process-wide memory after measured frames, not a peak or model-only measurement.

The CSV is one stable summary row intended for later comparison, with columns including implementation, mode, frames, warm-up frames, stage mean/percentiles, FPS, and RSS. It contains Python only; no C++ values or comparisons are claimed.
