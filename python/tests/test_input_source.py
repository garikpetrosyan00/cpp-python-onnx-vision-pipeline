from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vision_pipeline.config import parse_source
from vision_pipeline.input_source import InputSource, MediaError


def fake_capture(monkeypatch, frames=(), *, opened=True, fps=12.0, count=0):
    capture = MagicMock()
    capture.isOpened.return_value = opened
    capture.get.side_effect = lambda prop: fps if prop == cv2.CAP_PROP_FPS else count
    capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda: capture)
    return capture


def test_image_once_and_closed(image_path: Path) -> None:
    reader = InputSource(parse_source(str(image_path)))
    with reader as source:
        assert iter(source) is source
        np.testing.assert_array_equal(next(source), cv2.imread(str(image_path)))
        for _ in range(2):
            with pytest.raises(StopIteration):
                next(source)
    with pytest.raises(MediaError, match="closed"):
        next(reader)


def test_source_context_cannot_be_nested_and_can_be_reopened(image_path: Path) -> None:
    reader = InputSource(parse_source(str(image_path)))
    for _ in range(2):
        with reader as source:
            with pytest.raises(MediaError, match="already open"), reader:
                pass
            assert len(list(source)) == 1


def test_real_video_and_eof(video_path: Path) -> None:
    with InputSource(parse_source(str(video_path))) as source:
        assert source.fps == pytest.approx(12.0)
        frames = list(source)
        assert len(frames) == 6
        for index, frame in enumerate(frames):
            assert frame.shape == (48, 64, 3)
            assert float(frame.mean()) == pytest.approx(index * 30, abs=3)
        with pytest.raises(StopIteration):
            next(source)


@pytest.mark.parametrize(
    "suffix, message", [(".png", "Cannot read image"), (".avi", "Cannot open video")]
)
@pytest.mark.parametrize("data", [b"", b"not valid media"])
def test_unreadable_media(tmp_path: Path, suffix: str, message: str, data: bytes) -> None:
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(data)
    with pytest.raises(MediaError, match=message), InputSource(parse_source(str(path))):
        pass


def test_camera_frames_and_release(monkeypatch) -> None:
    frame = np.full((48, 64, 3), 75, dtype=np.uint8)
    capture = fake_capture(monkeypatch, [frame])
    with InputSource(parse_source("0")) as source:
        assert next(source) is frame
    capture.open.assert_called_once_with(0)
    capture.release.assert_called_once()


def test_unavailable_camera_released(monkeypatch) -> None:
    capture = fake_capture(monkeypatch, opened=False)
    with pytest.raises(MediaError, match="Camera 2 is unavailable"), InputSource(parse_source("2")):
        pass
    capture.release.assert_called_once()


def test_camera_read_failure_is_not_eof(monkeypatch) -> None:
    capture = fake_capture(monkeypatch)
    with (
        pytest.raises(MediaError, match="Cannot read a frame"),
        InputSource(parse_source("0")) as source,
    ):
        next(source)
    capture.release.assert_called_once()


@pytest.mark.parametrize(
    "count, frames, message",
    [(0, [], "empty"), (3, [np.zeros((2, 2, 3), dtype=np.uint8)], "truncated")],
)
def test_empty_or_damaged_video(
    monkeypatch, tmp_path: Path, count: int, frames, message: str
) -> None:
    path = tmp_path / "test.avi"
    path.touch()
    capture = fake_capture(monkeypatch, frames, count=count)
    with pytest.raises(MediaError, match=message), InputSource(parse_source(str(path))) as source:
        list(source)
    capture.release.assert_called_once()


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_falls_back(monkeypatch, fps: float) -> None:
    fake_capture(monkeypatch, fps=fps)
    with InputSource(parse_source("0")) as source:
        assert source.fps == 30.0


@pytest.mark.parametrize("operation", ["open", "read"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, cv2.error])
def test_capture_released_on_exception(monkeypatch, operation: str, error) -> None:
    capture = fake_capture(monkeypatch)
    getattr(capture, operation).side_effect = error("interrupted")
    expected = KeyboardInterrupt if error is KeyboardInterrupt else MediaError
    with pytest.raises(expected), InputSource(parse_source("0")) as source:
        next(source)
    capture.release.assert_called_once()
