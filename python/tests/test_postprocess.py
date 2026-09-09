import numpy as np
import pytest

from vision_pipeline.config import DEFAULT_LABELS, load_labels
from vision_pipeline.model_contract import OUTPUT_SHAPE, DetectorError
from vision_pipeline.postprocess import box_iou, decode_boxes, nms, postprocess
from vision_pipeline.preprocess import preprocess


@pytest.fixture
def raw():
    return np.zeros(OUTPUT_SHAPE, dtype=np.float32)


@pytest.fixture
def labels():
    return load_labels(DEFAULT_LABELS)


@pytest.fixture
def metadata():
    return preprocess(np.zeros((416, 416, 3), dtype=np.uint8)).metadata


def candidate(raw, anchor, class_id=0, center=(40, 40), objectness=1.0, class_score=0.5):
    # The tested anchor positions are all in stride-8's first row.
    raw[0, anchor, :4] = (center[0] / 8 - anchor, center[1] / 8, 0, 0)
    raw[0, anchor, 4] = objectness
    raw[0, anchor, 5 + class_id] = class_score


def test_raw_decode_all_level_boundaries_and_no_mutation(raw) -> None:
    before = raw.copy()
    decoded = decode_boxes(raw)
    indices = [0, 1, 52, 2703, 2704, 3379, 3380, 3548]
    expected = [
        [-4, -4, 4, 4],
        [4, -4, 12, 4],
        [-4, 4, 4, 12],
        [404, 404, 412, 412],
        [-8, -8, 8, 8],
        [392, 392, 408, 408],
        [-16, -16, 16, 16],
        [368, 368, 400, 400],
    ]
    np.testing.assert_array_equal(decoded[indices], expected)
    np.testing.assert_array_equal(raw, before)


def test_exponential_size_decode(raw) -> None:
    raw[0, 0, 2:4] = np.log(np.array([2, 4], dtype=np.float32))
    np.testing.assert_array_equal(decode_boxes(raw)[0], [-8, -16, 8, 16])


def test_pixel_inclusive_iou() -> None:
    boxes = np.array([[0, 0, 9, 9], [5, 0, 14, 9], [20, 0, 29, 9], [9, 0, 18, 9]], dtype=np.float32)
    np.testing.assert_allclose(box_iou(boxes[0], boxes), [1, 1 / 3, 0, 1 / 19], rtol=0, atol=2e-8)
    assert box_iou(boxes[0], np.empty((0, 4), dtype=np.float32)).shape == (0,)


def test_nms_order_and_boundary() -> None:
    boxes = np.array([[0, 0, 9, 9], [0, 0, 9, 9], [20, 0, 29, 9]], dtype=np.float32)
    scores = np.array([0.5, 0.5, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.45) == [2, 0]
    assert nms(boxes, scores, 1.0) == [2, 0, 1]  # Equality is not suppression.
    assert nms(boxes, scores, 0.0) == [2, 0]  # IoU=0 survives.
    assert nms(boxes[:0], scores[:0], 0.45) == []


def test_class_aware_suppression_and_multilabel(raw, labels, metadata) -> None:
    candidate(raw, 0, class_id=0, class_score=0.75)
    candidate(raw, 1, class_id=0, class_score=0.5)
    raw[0, 0, 5 + 1] = 0.625
    detections = postprocess(raw, metadata, labels)
    assert [(d.class_id, d.label, d.confidence) for d in detections] == [
        (0, "person", 0.75),
        (1, "bicycle", 0.625),
    ]
    assert [(d.x1, d.y1, d.x2, d.y2) for d in detections] == [(36, 36, 44, 44)] * 2


def test_threshold_inclusive_objectness_product(raw, labels, metadata) -> None:
    candidate(raw, 0, class_id=79, objectness=0.5, class_score=0.5)
    candidate(
        raw,
        1,
        center=(80, 80),
        objectness=0.5,
        class_score=np.nextafter(np.float32(0.5), np.float32(0)),
    )
    detections = postprocess(raw, metadata, labels, 0.25)
    assert len(detections) == 1
    assert detections[0].confidence == 0.25
    assert detections[0].label == "toothbrush"
    assert (
        postprocess(raw, metadata, labels, float(np.nextafter(np.float32(0.25), np.float32(1))))
        == []
    )


def test_global_nms_order_ties(raw, labels, metadata) -> None:
    candidate(raw, 0, class_id=5, center=(40, 40), class_score=0.5)
    candidate(raw, 1, class_id=0, center=(80, 40), class_score=0.5)
    candidate(raw, 2, class_id=0, center=(120, 40), class_score=0.5)
    candidate(raw, 3, class_id=79, center=(160, 40), class_score=0.75)
    result = postprocess(raw, metadata, labels)
    assert [d.class_id for d in result] == [79, 0, 0, 5]
    assert [d.x1 for d in result] == [156, 76, 116, 36]


def test_coordinate_restoration_and_bounds(raw, labels) -> None:
    metadata = preprocess(np.zeros((104, 208, 3), dtype=np.uint8)).metadata
    candidate(raw, 0, center=(40, 40), class_score=0.75)
    candidate(raw, 1, center=(0, 0))
    candidate(raw, 2, center=(416, 208))
    candidate(raw, 3, center=(800, 800))  # Entirely outside image; remove after clipping.
    result = postprocess(raw, metadata, labels)
    assert [(d.x1, d.y1, d.x2, d.y2) for d in result] == [
        (18, 18, 22, 22),
        (0, 0, 2, 2),
        (206, 102, 208, 104),
    ]


def test_no_detections(raw, metadata, labels) -> None:
    assert postprocess(raw, metadata, labels) == []


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((3549, 85), dtype=np.float32),
        np.zeros((1, 85, 3549), dtype=np.float32),
        np.zeros((1, 3549, 84), dtype=np.float32),
        np.zeros((2, 3549, 85), dtype=np.float32),
        np.zeros((1, 0, 85), dtype=np.float32),
        np.zeros(OUTPUT_SHAPE, dtype=np.float64),
        [],
    ],
)
def test_malformed_outputs(output, metadata, labels) -> None:
    with pytest.raises(DetectorError, match="expected float32"):
        postprocess(output, metadata, labels)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", [0, 4, 84])
def test_nonfinite_outputs(raw, metadata, labels, value, column) -> None:
    raw[0, 10, column] = value
    with pytest.raises(DetectorError, match="NaN or infinite"):
        postprocess(raw, metadata, labels)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probabilities_not_logits(raw, labels, metadata, value) -> None:
    raw[0, 0, 4] = value
    with pytest.raises(DetectorError, match="probabilities"):
        postprocess(raw, metadata, labels)


def test_decode_overflow(raw, labels, metadata) -> None:
    raw[0, 0, 2] = 100
    with pytest.raises(DetectorError, match="overflowed"):
        postprocess(raw, metadata, labels)


def test_wrong_label_count(raw, metadata, labels) -> None:
    with pytest.raises(DetectorError, match="80 labels"):
        postprocess(raw, metadata, labels[:-1])


def test_nms_rejects_arithmetic_overflow() -> None:
    boxes = np.array([[-1e30, -1e30, 1e30, 1e30]] * 2, dtype=np.float32)
    with pytest.raises(DetectorError, match="IoU arithmetic overflowed"):
        nms(boxes, np.array([0.8, 0.7], dtype=np.float32), 0.45)
