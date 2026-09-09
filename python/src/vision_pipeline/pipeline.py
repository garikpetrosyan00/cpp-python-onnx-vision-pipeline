"""Phase 1 passthrough; no model loading, inference, drawing, or metrics."""

from vision_pipeline.config import PipelineConfig
from vision_pipeline.input_source import InputSource
from vision_pipeline.render import Renderer


def run_pipeline(config: PipelineConfig) -> int:
    """Own all media resources and return the number of frames processed."""
    frames = 0
    with InputSource(config.source) as source, Renderer(config, source.fps) as renderer:
        for frame in source:
            frames += 1
            if not renderer.render(frame):
                break
            if config.max_frames is not None and frames >= config.max_frames:
                break
    return frames
