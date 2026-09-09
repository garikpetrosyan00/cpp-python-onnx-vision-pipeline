# Model Contract

## Planned Default

YOLOX-Nano is the current default strategy because it is compact, has an upstream ONNX Runtime example and pre-generated ONNX artifact, and keeps the detection math visible in both languages.

Planned baseline:

- Input: batch 1, float32 NCHW, `416x416`.
- Provider: ONNX Runtime CPU execution.
- Runtime file: `models/detector.onnx`, downloaded and checksum-verified locally.
- Runtime framework: ONNX Runtime directly; YOLOX/PyTorch are not runtime dependencies.

## Audit Required Before Phase 2 Implementation

- Exact upstream release URL and immutable SHA-256.
- Artifact and class-label provenance plus redistribution terms.
- ONNX opset, input/output names, shapes, and data types.
- Whether grids/strides are decoded in the graph or in postprocessing.
- Letterbox fill, scale/padding, channel order, normalization, and coordinate restoration.
- Confidence calculation and class-aware NMS semantics.

Upstream: <https://github.com/Megvii-BaseDetection/YOLOX>
