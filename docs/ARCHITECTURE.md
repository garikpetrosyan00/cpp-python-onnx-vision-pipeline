# Architecture

![CPU vision pipeline architecture](architecture.svg)

Both CLIs own media capture, rendering, and output lifecycles deterministically. Supplying `--model` activates one reusable ONNX Runtime CPU session before frame acquisition; omitting it preserves passthrough media behavior. The audited static YOLOX-Nano contract is shared: BGR values in `0..255`, top-left 416×416 letterbox padding value 114, contiguous float32 NCHW input, raw `[1,3549,85]` decoding, confidence filtering, class-aware NMS, coordinate restoration, and clamping.

Canonical detection JSON provides the Phase 6 comparison surface. Benchmark JSON and CSV capture Phase 7 timing aggregates; the comparative runner validates matching model, source, settings, and CPU provider before reporting. See [MODEL.md](MODEL.md), [PARITY.md](PARITY.md), and [BENCHMARKING.md](BENCHMARKING.md).
