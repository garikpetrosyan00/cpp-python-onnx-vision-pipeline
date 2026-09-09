# Models

Downloaded model files are local artifacts and are ignored by Git.

The planned Phase 2 default is the official YOLOX-Nano ONNX model with a `416x416` input. Before the download helper is finalized, the exact release URL, SHA-256 checksum, model-file provenance, license/redistribution notes, opset, tensor names, tensor shapes, preprocessing, and output decoding will be verified against the actual artifact.

Expected local paths:

```text
models/detector.onnx
models/classes.txt
```

Do not add third-party weights to the repository without a separate size and licensing decision.
