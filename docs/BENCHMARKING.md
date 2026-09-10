# Comparative CPU benchmarking (Phase 7)

Both CLIs support the same headless benchmark schema and are compared by the Phase 7 runner.

```bash
python/.venv/bin/python -m vision_pipeline \
  --model models/detector.onnx \
  --labels models/classes.txt \
  --source path/to/input.avi \
  --benchmark --no-display --warmup 5 --max-frames 200 \
  --benchmark-output benchmarks/results/python-run.json
```

`--benchmark` requires `--no-display`; it also works without `--model` to measure passthrough. `--warmup` is a non-negative integer and defaults to 5. Warm-up frames follow the full capture, detector (if enabled), annotation/output, and headless-render path, but contribute no samples. In benchmark mode `--max-frames` limits measured frames only, excluding warm-up. Without `--benchmark`, `--max-frames` retains its normal all-processed-frames meaning.

End-of-file before the first measured frame is an error. A clean run writes JSON to the explicit `--benchmark-output` or to `benchmarks/results/python-benchmark.json`, plus a sibling `.csv` file with the same base name. Both files are staged in the destination directory, then published together; failed collection or publication preserves existing results. Ctrl+C never publishes a benchmark result, even if it finalizes already-written annotated media.

All timing values begin as raw integer nanoseconds. `capture_ms` measures only `InputSource.next(frame)`. `preprocess_ms` is BGR frame to tensor; `inference_ms` is only `InferenceEngine.run`; `postprocess_ms` covers decode/filter/NMS/restoration/clamping; `render_ms` covers annotation, output writing, and GUI drawing. GUI wait time is excluded. `total_ms` is exactly the sum of those processing boundaries, and effective FPS is `measured_frames * 1e9 / sum(total_ns)`. Passthrough reports exactly zero preprocess, inference, and postprocess timing.

Each timing summary has count, mean, median/p50, p95, p99, minimum, and maximum in milliseconds. Percentiles use linear interpolation between sorted samples at index `(n - 1) * p / 100`; p50 is the median. An empty set has count zero and null summary/FPS values. The JSON records schema version, mode, source kind, requested/completed warm-up and measured counts, thresholds, display setting, model/provider identity when applicable, dependency versions, all timing definitions/summaries, and a current-process RSS snapshot. RSS is approximate process-wide memory after measured frames, not a peak or model-only measurement.

The CSV is one stable summary row with implementation, mode, frames, warm-up frames, stage mean/percentiles, FPS, and RSS. Use the runner to validate model SHA, CPU provider, thresholds, counts, JSON/CSV schema, and source metadata before comparison:

```bash
python/.venv/bin/python benchmarks/run_benchmarks.py \
  --python python/.venv/bin/python --cpp /tmp/vision-phase7-build/vision_cpp \
  --model models/detector.onnx --labels models/classes.txt \
  --source assets/sample/benchmark.avi --frames 30 --warmup 5 \
  --results-dir /tmp/vision-phase7-results
```

The tracked source is a project-generated 416x416, 15 FPS, 40-frame MJPG AVI. Its deterministic BGR coordinate gradient and moving primitives contain no third-party content. SHA-256: `5817f0596a8d6b1abedf0622bdee969fc0114e4f8d492fe9acdafed75ec7629b`.

RSS is an approximate current process-wide snapshot after measured frames, not peak, model-only, or comparable memory attribution. Results are host-, OpenCV codec-, and ONNX Runtime-dependent.
