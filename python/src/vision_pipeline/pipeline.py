"""Media passthrough or acquire → preprocess → infer → postprocess → render."""

from contextlib import ExitStack

from vision_pipeline.config import PipelineConfig
from vision_pipeline.input_source import InputSource
from vision_pipeline.postprocess import postprocess
from vision_pipeline.preprocess import preprocess
from vision_pipeline.render import Renderer


def run_pipeline(config: PipelineConfig) -> int:
    """Validate the detector before acquisition; own all resources until exit."""
    frames = 0
    with ExitStack() as resources:
        detector = None
        if config.model is not None:
            from vision_pipeline.inference import InferenceEngine

            detector = resources.enter_context(InferenceEngine(config.model))
        source = resources.enter_context(InputSource(config.source))
        renderer = resources.enter_context(Renderer(config, source.fps))
        for frame in source:
            detections = []
            if detector is not None:
                prepared = preprocess(frame)
                output = detector.run(prepared.tensor)
                detections = postprocess(
                    output, prepared.metadata, config.label_names, config.confidence, config.iou
                )
            frames += 1
            if not renderer.render(frame, detections):
                break
            if config.max_frames is not None and frames >= config.max_frames:
                break
    return frames
