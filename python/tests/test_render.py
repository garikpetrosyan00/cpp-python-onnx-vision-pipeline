from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vision_pipeline.postprocess import Detection
from vision_pipeline.render import annotate, class_color


def test_annotations_preserve_input_and_draw_class_confidence(monkeypatch) -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    before = frame.copy()
    text = MagicMock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", text)
    detection = Detection(0, "person", 0.75, 10, 20, 70, 60)
    result = annotate(frame, [detection])
    assert not np.shares_memory(frame, result)
    np.testing.assert_array_equal(frame, before)
    assert np.any(result != frame)
    assert text.call_args.args[1] == "person 0.75"
    np.testing.assert_array_equal(result[40, 10], class_color(0))
    assert class_color(0) == class_color(0)
    assert class_color(0) != class_color(1)


@pytest.mark.parametrize("size", [(80, 120), (1, 1), (2, 3)])
def test_all_drawing_coordinates_clamped(monkeypatch, size) -> None:
    rectangle = MagicMock()
    text = MagicMock()
    monkeypatch.setattr(cv2, "rectangle", rectangle)
    monkeypatch.setattr(cv2, "putText", text)
    frame = np.zeros((*size, 3), dtype=np.uint8)
    annotate(frame, [Detection(1, "bicycle", 0.8, -100, -100, 1000, 1000)])
    for call in rectangle.call_args_list:
        for x, y in call.args[1:3]:
            assert 0 <= x < size[1] and 0 <= y < size[0]
    for call in text.call_args_list:
        x, y = call.args[2]
        assert 0 <= x < size[1] and 0 <= y < size[0]


def test_empty_detections_keep_original_frame() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert annotate(frame, []) is frame
