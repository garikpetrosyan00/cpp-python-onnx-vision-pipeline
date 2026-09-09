# Architecture

## Planned Pipeline

```text
InputSource -> Preprocessor -> InferenceEngine -> Postprocessor
            -> Renderer -> Output -> MetricsCollector
```

Python and C++ will keep equivalent configuration, preprocessing, postprocessing, coordinate restoration, rendering semantics, and timing boundaries. Phase 0 contains only help/version entry points; module details will be added with the phases that implement them.
