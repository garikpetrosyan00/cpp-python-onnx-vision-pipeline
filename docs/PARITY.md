# Phase 6 parity

Phase 6 is complete. This document records the bounded, reproducible evidence used by the finished Phase 0–8 core; it does not claim general cross-model or cross-platform equivalence.

`assets/sample/parity.png` is a project-generated, tracked 416x416 PNG containing a deterministic BGR coordinate gradient. With zero-based pixel coordinates `(x, y)`, its OpenCV channel values are `B=(3x+y) mod 256`, `G=(x+2y) mod 256`, and `R=(5x+7y) mod 256`. It was generated as a NumPy `uint8` array and encoded losslessly with `cv2.imwrite`; it contains no third-party content. Its SHA-256 is `1f1fd7597addb87ab7273dd646192c33efcd28d9b3bfc72db477803b2135b10c`. Its exact model-size dimensions avoid resize-version differences.

Both CLIs write `vision-pipeline-detections/v1`: schema/implementation/model contract identity, image dimensions, thresholds, and deterministic detections. Each detection has only `class_id`, `label`, `confidence`, `x1`, `y1`, `x2`, and `y2`. Publication is atomic after a successful image run.

Run `python/.venv/bin/python scripts/check_parity.py --python python/.venv/bin/python --cpp /tmp/vision-phase6-final-build/vision_cpp --model models/detector.onnx --labels models/classes.txt --image assets/sample/parity.png --work-dir /tmp/vision-phase6-parity`.

The script requires a nonzero exact count, ordered class IDs, and labels. Float fields use an absolute tolerance of `1e-5`, appropriate for float32 tensor output and equivalent restoration arithmetic; it reports confidence and coordinate maxima separately and prints the exact differing record/field otherwise. C++ reproduces NumPy's vector float32 exponential with the same Cody-Waite reduction, FMA polynomial, and float32 assignment points because the platform `expf` gives different last bits. This proves only this audited image/model/configuration path, not performance or general parity.
