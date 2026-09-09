"""YOLOX non-legacy BGR preprocessing; see docs/MODEL.md and models/LICENSE.YOLOX.

Follows Megvii's preproc algorithm (Copyright (c) Megvii, Inc. and its affiliates),
with typed metadata and explicit validation added for this pipeline.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.input_source import Frame
from vision_pipeline.model_contract import INPUT_SHAPE, DetectorError


@dataclass(frozen=True)
class ResizeMetadata:
    """Sizes are (height, width); padding is (left, top, right, bottom)."""

    original_size: tuple[int, int]
    model_size: tuple[int, int]
    resized_size: tuple[int, int]
    ratio: float
    padding: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessedFrame:
    tensor: NDArray[np.float32]
    metadata: ResizeMetadata


def preprocess(frame: Frame) -> PreprocessedFrame:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
    ):
        raise DetectorError("Preprocessing expects a nonempty HxWx3 uint8 BGR frame.")
    height, width = frame.shape[:2]
    model_height, model_width = INPUT_SHAPE[2:]
    ratio = min(model_height / height, model_width / width)
    resized_width, resized_height = int(width * ratio), int(height * ratio)
    if min(resized_width, resized_height) < 1:
        raise DetectorError(
            f"Frame {width}x{height} is too narrow for the audited resize: "
            "one dimension rounds to zero. Use a less extreme aspect ratio."
        )
    try:
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    except cv2.error as exc:
        raise DetectorError(f"Cannot resize frame {width}x{height} for YOLOX-Nano: {exc}") from exc
    canvas = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return PreprocessedFrame(
        tensor,
        ResizeMetadata(
            (height, width),
            (model_height, model_width),
            (resized_height, resized_width),
            ratio,
            (0, 0, model_width - resized_width, model_height - resized_height),
        ),
    )
