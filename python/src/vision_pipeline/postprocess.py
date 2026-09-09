"""Raw YOLOX decode and class-aware NMS, with deterministic project boundaries.

Algorithms follow Megvii's demo_utils.py (Copyright (c) Megvii Inc.), Apache-2.0.
See docs/MODEL.md and models/LICENSE.YOLOX for attribution and intentional changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import unit_interval
from vision_pipeline.model_contract import CLASS_COUNT, INPUT_SHAPE, OUTPUT_SHAPE, DetectorError
from vision_pipeline.preprocess import ResizeMetadata

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def validate_output(output: FloatArray) -> None:
    if (
        not isinstance(output, np.ndarray)
        or output.shape != OUTPUT_SHAPE
        or output.dtype != np.float32
    ):
        raise DetectorError(
            f"Unexpected model output: shape={getattr(output, 'shape', None)}, "
            f"dtype={getattr(output, 'dtype', None)}; expected float32 {OUTPUT_SHAPE} "
            "with 80 class probabilities and raw YOLOX boxes."
        )
    if not np.isfinite(output).all():
        raise DetectorError(
            "Model output contains NaN or infinite values; expected finite YOLOX data."
        )
    if np.any(output[..., 4:] < 0) or np.any(output[..., 4:] > 1):
        raise DetectorError(
            "Model objectness/class probabilities must be in [0, 1]; no logits expected."
        )


def decode_boxes(output: FloatArray) -> FloatArray:
    """Decode raw center/size fields without mutating session output."""
    validate_output(output)
    grids = []
    scales = []
    for stride in (8, 16, 32):
        # Integer grids preserve the upstream NumPy promotion/assignment sequence.
        x, y = np.meshgrid(np.arange(INPUT_SHAPE[3] // stride), np.arange(INPUT_SHAPE[2] // stride))
        grid = np.stack((x, y), axis=-1).reshape(-1, 2)
        grids.append(grid)
        scales.append(np.full((len(grid), 1), stride))
    grid = np.concatenate(grids)
    strides = np.concatenate(scales)
    boxes = output[0, :, :4].copy()
    try:
        with np.errstate(over="raise", invalid="raise"):
            boxes[:, :2] = (boxes[:, :2] + grid) * strides
            boxes[:, 2:] = np.exp(boxes[:, 2:]) * strides
            corners = np.empty_like(boxes)
            corners[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            corners[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
    except FloatingPointError as exc:
        raise DetectorError(
            "YOLOX box decoding overflowed; check the model/output contract."
        ) from exc
    return corners


def box_iou(box: FloatArray, boxes: FloatArray) -> FloatArray:
    """Upstream pixel-inclusive (+1) IoU for one corner box against N corner boxes."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            intersection_size = np.maximum(
                0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]) + 1
            )
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            area = np.prod(np.maximum(0, box[2:] - box[:2] + 1))
            sizes = np.maximum(0, boxes[:, 2:] - boxes[:, :2] + 1)
            union = area + sizes[:, 0] * sizes[:, 1] - intersection
            return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    except FloatingPointError as exc:
        raise DetectorError(
            "NMS IoU arithmetic overflowed; check the model/output contract."
        ) from exc


def nms(boxes: FloatArray, scores: FloatArray, iou_threshold: float) -> list[int]:
    """Descending score, stable input-index ties; suppress only IoU > threshold."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        order = remaining[box_iou(boxes[index], boxes[remaining]) <= iou_threshold]
    return keep


def postprocess(
    output: FloatArray,
    metadata: ResizeMetadata,
    labels: Sequence[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    confidence_threshold = unit_interval(confidence_threshold, "--confidence")
    iou_threshold = unit_interval(iou_threshold, "--iou")
    if len(labels) != CLASS_COUNT:
        raise DetectorError(
            f"Expected {CLASS_COUNT} labels for YOLOX-Nano, observed {len(labels)}."
        )
    if (
        metadata.model_size != INPUT_SHAPE[2:]
        or not np.isfinite(metadata.ratio)
        or metadata.ratio <= 0
        or min(metadata.original_size) <= 0
    ):
        raise DetectorError(
            "Invalid coordinate restoration metadata for the 416x416 YOLOX contract."
        )
    boxes = decode_boxes(output)
    left, top, _, _ = metadata.padding
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / metadata.ratio
            boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / metadata.ratio
    except FloatingPointError as exc:
        raise DetectorError(
            "Restored box coordinates overflowed; check the frame/model contract."
        ) from exc
    scores = output[0, :, 4:5] * output[0, :, 5:]
    height, width = metadata.original_size
    results: list[tuple[Detection, int]] = []
    for class_id in range(CLASS_COUNT):
        anchors = np.flatnonzero(scores[:, class_id] >= confidence_threshold)
        class_scores = scores[anchors, class_id]
        for selected in nms(boxes[anchors], class_scores, iou_threshold):
            anchor = int(anchors[selected])
            x1, y1, x2, y2 = np.clip(boxes[anchor], (0, 0, 0, 0), (width, height, width, height))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                (
                    Detection(
                        class_id,
                        labels[class_id],
                        float(class_scores[selected]),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                    anchor,
                )
            )
    results.sort(key=lambda item: (-item[0].confidence, item[0].class_id, item[1]))
    return [detection for detection, _ in results]
