# Phase 7 benchmark report

Measured on this Ubuntu host using `assets/sample/benchmark.avi` (416x416, 15 FPS, 40 frames, MJPG; SHA-256 `5817f0596a8d6b1abedf0622bdee969fc0114e4f8d492fe9acdafed75ec7629b`), audited YOLOX-Nano SHA-256 `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`, confidence `0.25`, IoU `0.45`, CPUExecutionProvider, 5 warm-up frames, and 30 measured frames. Raw validated documents are in `/tmp/vision-phase7-results/` for this local run.

```bash
python/.venv/bin/python benchmarks/run_benchmarks.py --python python/.venv/bin/python --cpp /tmp/vision-phase7-build/vision_cpp --model models/detector.onnx --labels models/classes.txt --source assets/sample/benchmark.avi --frames 30 --warmup 5 --results-dir /tmp/vision-phase7-results
```

| Implementation | Capture mean ms | Preprocess mean ms | Inference mean / p50 / p95 ms | Postprocess mean ms | Render mean ms | Total mean / p50 / p95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 2.093 | 0.880 | 26.654 / 26.679 / 30.097 | 4.517 | 0.050 | 34.193 / 34.227 / 38.019 | 29.245 |
| C++ | 1.944 | 0.758 | 29.117 / 29.959 / 32.043 | 3.214 | 0.001 | 35.033 / 35.416 / 38.524 | 28.544 |

Runtime details recorded by the documents: Python 3.12.14, NumPy 2.5.3, OpenCV 4.14.0, ONNX Runtime 1.24.4; C++17, OpenCV 4.5.4, ONNX Runtime 1.24.4. The runner validated matching model SHA, CPU provider, thresholds, source kind, warm-up/measured counts, JSON schema, and CSV structure before producing this table.

These are one host-specific measured run, not a claim of universal performance. Codec, OpenCV build, ONNX Runtime build, CPU scheduling, and thermal state can change them. RSS snapshots were Python 125.587 MB and C++ 121.221 MB; they are current process-wide snapshots after measurement, not peak or model-only memory figures.
