"""End-to-end output and lifecycle tests, with camera and GUI calls mocked."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vision_pipeline.cli import main
from vision_pipeline.config import PipelineConfig, parse_source
from vision_pipeline.input_source import InputSource, MediaError
from vision_pipeline.pipeline import run_pipeline
from vision_pipeline.render import Renderer


@pytest.fixture
def gui(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":mock")
    calls = MagicMock()
    for name in ("namedWindow", "imshow", "waitKey", "destroyWindow"):
        monkeypatch.setattr(cv2, name, getattr(calls, name))
    calls.waitKey.return_value = ord("q")
    return calls


def test_headless_image_roundtrip(image_path: Path, tmp_path: Path, gui) -> None:
    output = tmp_path / "nested" / "output.png"
    assert (
        main(
            [
                "--source",
                str(image_path),
                "--output",
                str(output),
                "--no-display",
                "--max-frames",
                "1",
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(cv2.imread(str(image_path)), cv2.imread(str(output)))
    assert gui.mock_calls == []
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("limit, expected", [(None, 6), (2, 2), (20, 6)])
def test_video_output(video_path: Path, tmp_path: Path, limit: int | None, expected: int) -> None:
    output = tmp_path / "nested" / "output.avi"
    config = PipelineConfig(parse_source(str(video_path)), output, True, limit)
    assert run_pipeline(config) == expected
    with InputSource(parse_source(str(output))) as source:
        assert source.fps == pytest.approx(12.0)
        frames = list(source)
    assert len(frames) == expected
    for index, frame in enumerate(frames):
        assert frame.shape == (48, 64, 3)
        assert float(frame.mean()) == pytest.approx(index * 30, abs=5)
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
@pytest.mark.parametrize("kind", ["image", "video"])
def test_interactive_quit(image_path: Path, video_path: Path, gui, key: int, kind: str) -> None:
    gui.waitKey.return_value = key
    path = image_path if kind == "image" else video_path
    assert run_pipeline(PipelineConfig(parse_source(str(path)))) == 1
    gui.namedWindow.assert_called_once()
    if kind == "image":
        np.testing.assert_array_equal(gui.imshow.call_args.args[1], cv2.imread(str(image_path)))
    gui.destroyWindow.assert_called_once()


def test_image_waits_for_exit_and_polls(image_path: Path, gui) -> None:
    gui.waitKey.side_effect = [-1, -1, 27]
    assert run_pipeline(PipelineConfig(parse_source(str(image_path)))) == 1
    assert gui.waitKey.call_count == 3
    assert all(call.args[0] > 0 for call in gui.waitKey.call_args_list)


def test_no_display_session(image_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vision_pipeline.render.sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(MediaError, match="--no-display"):
        run_pipeline(PipelineConfig(parse_source(str(image_path))))


def test_gui_error_closes_window(image_path: Path, gui, capsys) -> None:
    gui.imshow.side_effect = cv2.error("GUI unavailable")
    assert main(["--source", str(image_path)]) == 1
    assert "--no-display" in capsys.readouterr().err
    gui.destroyWindow.assert_called_once()


def test_ctrl_c_preserves_completed_video(video_path: Path, tmp_path: Path, gui, capsys) -> None:
    output = tmp_path / "interrupted.avi"
    gui.waitKey.side_effect = [-1, KeyboardInterrupt]
    assert main(["--source", str(video_path), "--output", str(output)]) == 130
    assert "Interrupted" in capsys.readouterr().err
    gui.destroyWindow.assert_called_once()
    with InputSource(parse_source(str(output))) as source:
        assert len(list(source)) == 2
    assert not list(tmp_path.glob(".interrupted-*"))


def test_camera_max_frames_and_video_output(tmp_path: Path, monkeypatch) -> None:
    real_capture = cv2.VideoCapture
    camera = MagicMock()
    camera.isOpened.return_value = True
    camera.get.return_value = 0.0  # No reported FPS; output should use 30.
    camera.read.side_effect = [
        (True, np.full((48, 64, 3), index * 30, dtype=np.uint8)) for index in range(4)
    ]
    # First capture is the camera, second verifies encoded output.
    factory = MagicMock(side_effect=[camera, real_capture()])
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    output = tmp_path / "camera.avi"
    assert run_pipeline(PipelineConfig(parse_source("0"), output, True, 3)) == 3
    assert camera.read.call_count == 3
    camera.release.assert_called_once()
    monkeypatch.setattr(cv2, "VideoCapture", real_capture)
    with InputSource(parse_source(str(output))) as source:
        assert source.fps == pytest.approx(30.0)
        assert len(list(source)) == 3


@pytest.mark.parametrize("failure", [False, cv2.error("encoder failed")])
def test_image_write_failure_preserves_existing_output(
    image_path: Path, tmp_path: Path, monkeypatch, failure
) -> None:
    output = tmp_path / "output.png"
    output.write_bytes(b"existing output")
    writer = MagicMock()
    if failure is False:
        writer.return_value = False
    else:
        writer.side_effect = failure
    monkeypatch.setattr(cv2, "imwrite", writer)
    with pytest.raises(MediaError, match="Cannot write"):
        run_pipeline(PipelineConfig(parse_source(str(image_path)), output, True))
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".output-*"))


def test_unwritable_parent(image_path: Path, tmp_path: Path, monkeypatch) -> None:
    def denied(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr("vision_pipeline.render.tempfile.NamedTemporaryFile", denied)
    with pytest.raises(MediaError, match="permissions"):
        run_pipeline(PipelineConfig(parse_source(str(image_path)), tmp_path / "out.png", True))


@pytest.mark.parametrize("opened", [False, True])
def test_unavailable_or_silently_failing_encoder(
    video_path: Path, tmp_path: Path, monkeypatch, opened: bool
) -> None:
    writer = MagicMock()
    writer.isOpened.return_value = opened
    monkeypatch.setattr(cv2, "VideoWriter", lambda: writer)
    output = tmp_path / "output.avi"
    message = "verification failed" if opened else "codec MJPG"
    with pytest.raises(MediaError, match=message):
        run_pipeline(PipelineConfig(parse_source(str(video_path)), output, True))
    writer.release.assert_called_once()
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))


def test_failure_to_publish_output(image_path: Path, tmp_path: Path, monkeypatch) -> None:
    def denied(*args):
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "replace", denied)
    output = tmp_path / "out.png"
    with pytest.raises(MediaError, match="Cannot finalize output"):
        run_pipeline(PipelineConfig(parse_source(str(image_path)), output, True))
    assert not output.exists()
    assert not list(tmp_path.glob(".out-*"))


def test_writer_exception_releases_writer_and_window(
    video_path: Path, tmp_path: Path, monkeypatch, gui
) -> None:
    writer = MagicMock()
    writer.isOpened.return_value = True
    writer.write.side_effect = [None, cv2.error("disk full")]
    gui.waitKey.return_value = -1
    monkeypatch.setattr(cv2, "VideoWriter", lambda: writer)
    with pytest.raises(MediaError, match="Cannot write output"):
        run_pipeline(PipelineConfig(parse_source(str(video_path)), tmp_path / "out.avi"))
    writer.release.assert_called_once()
    gui.destroyWindow.assert_called_once()
    assert not list(tmp_path.glob("*out*"))


def test_video_dimension_changes_are_reported(tmp_path: Path) -> None:
    config = PipelineConfig(parse_source("0"), tmp_path / "out.avi", True)
    with (
        pytest.raises(MediaError, match="changing frame dimensions"),
        Renderer(config, 30.0) as renderer,
    ):
        renderer.render(np.zeros((48, 64, 3), dtype=np.uint8))
        renderer.render(np.zeros((50, 64, 3), dtype=np.uint8))
    assert not config.output.exists()


def test_odd_video_dimensions_are_not_silently_cropped(tmp_path: Path) -> None:
    config = PipelineConfig(parse_source("0"), tmp_path / "out.avi", True)
    with (
        pytest.raises(MediaError, match="even frame dimensions"),
        Renderer(config, 30.0) as renderer,
    ):
        renderer.render(np.zeros((47, 63, 3), dtype=np.uint8))
    assert not config.output.exists()


def test_media_error_cli(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.png"
    path.touch()
    assert main(["--source", str(path), "--no-display"]) == 1
    error = capsys.readouterr().err
    assert "Cannot read image" in error
    assert "Traceback" not in error
