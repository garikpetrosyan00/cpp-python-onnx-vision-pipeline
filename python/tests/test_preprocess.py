import cv2
import numpy as np
import pytest

from vision_pipeline.model_contract import DetectorError
from vision_pipeline.preprocess import preprocess


@pytest.mark.parametrize(
    "original, resized, ratio, padding",
    [
        ((200, 400), (208, 416), 1.04, (0, 0, 0, 208)),
        ((400, 200), (416, 208), 1.04, (0, 0, 208, 0)),
        ((416, 416), (416, 416), 1.0, (0, 0, 0, 0)),
        ((1, 1), (416, 416), 416.0, (0, 0, 0, 0)),
        ((1, 2), (208, 416), 208.0, (0, 0, 0, 208)),
        ((123, 321), (159, 416), 416 / 321, (0, 0, 0, 257)),
        ((333, 101), (416, 126), 416 / 333, (0, 0, 290, 0)),
    ],
)
def test_exact_layout_color_values_and_metadata(original, resized, ratio, padding) -> None:
    frame = np.empty((*original, 3), dtype=np.uint8)
    frame[:] = (5, 77, 241)  # Deliberately distinct B/G/R values, far above 1.
    before = frame.copy()
    result = preprocess(frame)
    assert result.tensor.shape == (1, 3, 416, 416)
    assert result.tensor.dtype == np.float32
    assert result.tensor.flags.c_contiguous
    meta = result.metadata
    assert meta.original_size == original
    assert meta.model_size == (416, 416)
    assert meta.resized_size == resized
    assert meta.ratio == ratio
    assert meta.padding == padding
    for channel, value in enumerate((5, 77, 241)):
        assert np.all(result.tensor[0, channel, : resized[0], : resized[1]] == value)
    assert np.all(result.tensor[0, :, resized[0] :, :] == 114)
    assert np.all(result.tensor[0, :, :, resized[1] :] == 114)
    np.testing.assert_array_equal(frame, before)
    assert not np.shares_memory(frame, result.tensor)


def test_spatial_order_without_resizing() -> None:
    y, x = np.indices((416, 416))
    frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    result = preprocess(frame)
    for row, column in [(0, 0), (0, 415), (100, 201), (415, 415)]:
        np.testing.assert_array_equal(result.tensor[0, :, row, column], frame[row, column])


def test_uint8_bilinear_rounding_matches_opencv() -> None:
    frame = np.array(
        [[[0, 50, 100], [200, 250, 10]], [[40, 70, 120], [240, 90, 30]]], dtype=np.uint8
    )
    expected = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
    result = preprocess(frame)
    np.testing.assert_array_equal(result.tensor[0].transpose(1, 2, 0), expected)
    # Bilinear interpolation took place while values were uint8, before float32 conversion.
    assert np.equal(result.tensor, np.floor(result.tensor)).all()


def test_noncontiguous_frame() -> None:
    frame = np.zeros((40, 100, 3), dtype=np.uint8)[:, ::2]
    assert not frame.flags.c_contiguous
    assert preprocess(frame).tensor.flags.c_contiguous


@pytest.mark.parametrize(
    "frame",
    [
        None,
        [],
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 0, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
    ],
)
def test_invalid_frames(frame) -> None:
    with pytest.raises(DetectorError, match="HxWx3 uint8 BGR"):
        preprocess(frame)


def test_extreme_aspect_ratio_does_not_silently_change_resize_policy() -> None:
    with pytest.raises(DetectorError, match="rounds to zero"):
        preprocess(np.zeros((1, 1000, 3), dtype=np.uint8))
