# Portfolio summary

This repository demonstrates delivery of the same bounded computer-vision pipeline in Python and C++: media lifecycle handling, an audited ONNX Runtime CPU detector, deterministic postprocessing, cross-language result comparison, and reproducible benchmark reporting.

## Responsibilities demonstrated

- Designed independent Python and C++ CLIs for images, videos, cameras, headless operation, output staging, and clean resource release.
- Audited and pinned one YOLOX-Nano ONNX artifact rather than accepting arbitrary model exports; documented tensor metadata, preprocessing, decoding, provenance, and limits.
- Implemented the detector’s BGR letterbox, raw grid/stride decode, confidence rule, class-aware NMS, restoration, and labeled rendering in both languages.
- Added canonical detection JSON, a fixed legal input, and a runner that proves the documented Phase 6 path within `1e-5` absolute tolerance.
- Added matching benchmark boundaries, JSON/CSV results, a compatibility-validating comparative runner, and an honest host-specific report.

## Evidence

- The tracked parity fixture produced 32 ordered detections in the documented real Python/C++ run; see [PARITY.md](PARITY.md).
- The tracked 40-frame benchmark fixture and raw command produced the measured values in [benchmarks/REPORT.md](../benchmarks/REPORT.md).
- Python tests, C++ CTest, parity, and benchmark commands are documented in the README and plan. Test fixtures are local and model-independent unless explicitly marked as real-model checks.

## Technical decisions and limits

The completed core is intentionally CPU-only and supports only the audited static YOLOX-Nano contract. It does not include CUDA, Docker, cloud services, batching, asynchronous pipelines, web APIs, additional detectors, or broad performance claims. Codec, GUI, camera, and benchmark behavior depend on the local environment. Original repository code is Apache-2.0; see the root [LICENSE](../LICENSE) and [NOTICE](../NOTICE). The upstream source is also Apache-2.0, but the ONNX weight artifact has no separately verified weight license/model card; weights are downloaded locally and are not redistributed.

Phase 9 optional extensions have not started.
