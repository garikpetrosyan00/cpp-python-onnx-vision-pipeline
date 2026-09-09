"""End-to-end output and lifecycle tests, with camera and GUI calls mocked."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vision_pipeline.cli import main
from vision_pipeline.config import PipelineConfig, parse_source
from vision_pipeline.input_source import InputSource, MediaError
from vision_pipeline.metrics import BenchmarkError
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


def test_detector_session_once_and_annotated_video(
    video_path: Path, model_path: Path, tmp_path: Path, mocked_session
) -> None:
    session, constructor = mocked_session
    raw = session.run.return_value[0]
    raw[0, 0, :5] = (10, 10, 2, 2, 1)
    raw[0, 0, 5] = 0.75
    output = tmp_path / "detected.avi"
    config = PipelineConfig(parse_source(str(video_path)), output, True, 3, model=model_path)
    assert run_pipeline(config) == 3
    constructor.assert_called_once()
    assert session.run.call_count == 3
    with InputSource(parse_source(str(output))) as source:
        frames = list(source)
    assert len(frames) == 3
    assert np.any(frames[0] > 10)  # Original first frame was black; annotations were saved.


def test_no_detections_saves_unchanged_image(
    image_path: Path, model_path: Path, tmp_path: Path, mocked_session
) -> None:
    output = tmp_path / "no_detections.png"
    assert (
        main(
            [
                "--source",
                str(image_path),
                "--model",
                str(model_path),
                "--no-display",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(cv2.imread(str(output)), cv2.imread(str(image_path)))


def test_corrupt_model_fails_before_media(model_path: Path, monkeypatch, capsys) -> None:
    forbidden = MagicMock(side_effect=AssertionError("Media opened too early"))
    monkeypatch.setattr("vision_pipeline.pipeline.InputSource", forbidden)
    assert main(["--source", "0", "--model", str(model_path), "--no-display"]) == 1
    forbidden.assert_not_called()
    message = capsys.readouterr().err
    assert "checksum/size mismatch" in message and "CPUExecutionProvider" in message
    assert "Traceback" not in message


def test_missing_model_config_error_before_media(tmp_path: Path, monkeypatch, capsys) -> None:
    forbidden = MagicMock(side_effect=AssertionError("Media opened too early"))
    monkeypatch.setattr("vision_pipeline.pipeline.InputSource", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", "--model", str(tmp_path / "missing.onnx")])
    assert error.value.code == 2
    forbidden.assert_not_called()
    assert "--model" in capsys.readouterr().err


def test_metadata_error_before_media(model_path: Path, mocked_session, monkeypatch, capsys) -> None:
    session, _ = mocked_session
    session.get_outputs.return_value[0].shape = [1, 8400, 84]
    forbidden = MagicMock(side_effect=AssertionError("Media opened too early"))
    monkeypatch.setattr("vision_pipeline.pipeline.InputSource", forbidden)
    assert main(["--source", "0", "--model", str(model_path), "--no-display"]) == 1
    forbidden.assert_not_called()
    assert "8400" in capsys.readouterr().err


def test_inference_interrupt_finalizes_completed_frames(
    video_path: Path, model_path: Path, tmp_path: Path, mocked_session, capsys
) -> None:
    session, _ = mocked_session
    session.run.side_effect = [session.run.return_value, KeyboardInterrupt()]
    output = tmp_path / "interrupt.avi"
    assert (
        main(
            [
                "--source",
                str(video_path),
                "--model",
                str(model_path),
                "--no-display",
                "--output",
                str(output),
            ]
        )
        == 130
    )
    with InputSource(parse_source(str(output))) as source:
        assert len(list(source)) == 1
    assert "Interrupted" in capsys.readouterr().err


def test_threshold_flags_control_detection(
    image_path: Path, model_path: Path, tmp_path: Path, mocked_session
) -> None:
    session, _ = mocked_session
    raw = session.run.return_value[0]
    raw[0, 0, :5] = (10, 10, 2, 2, 1)
    raw[0, 0, 5] = 0.5
    output = tmp_path / "threshold.png"
    arguments = [
        "--source",
        str(image_path),
        "--model",
        str(model_path),
        "--output",
        str(output),
        "--no-display",
    ]
    assert main([*arguments, "--confidence", "0.5", "--iou", "1"]) == 0
    assert np.any(cv2.imread(str(output)) != cv2.imread(str(image_path)))
    assert main([*arguments, "--confidence", "0.6", "--iou", "0"]) == 0
    np.testing.assert_array_equal(cv2.imread(str(output)), cv2.imread(str(image_path)))


def test_iou_flag_controls_same_class_suppression(
    image_path: Path, model_path: Path, mocked_session, monkeypatch
) -> None:
    session, _ = mocked_session
    raw = session.run.return_value[0]
    raw[0, 0, :5] = (10, 10, 2, 2, 1)
    raw[0, 1, :5] = (9, 10, 2, 2, 1)  # Same decoded box at the next grid column.
    raw[0, :2, 5] = 0.75
    counts = []

    def capture_detections(frame, detections):
        counts.append(len(detections))
        return frame

    monkeypatch.setattr("vision_pipeline.render.annotate", capture_detections)
    for threshold in ("0.45", "1"):
        assert (
            main(
                [
                    "--model",
                    str(model_path),
                    "--source",
                    str(image_path),
                    "--no-display",
                    "--iou",
                    threshold,
                ]
            )
            == 0
        )
    assert counts == [1, 2]


def test_inference_failure_preserves_existing_output(
    video_path: Path, model_path: Path, tmp_path: Path, mocked_session, capsys
) -> None:
    session, _ = mocked_session
    session.run.side_effect = [session.run.return_value, RuntimeError("inference failed")]
    output = tmp_path / "existing.avi"
    output.write_bytes(b"existing output")
    assert (
        main(
            [
                "--source",
                str(video_path),
                "--model",
                str(model_path),
                "--no-display",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".existing-*"))
    assert "inference failed" in capsys.readouterr().err


def test_passthrough_benchmark_warmup_and_measured_limit(video_path: Path, tmp_path: Path) -> None:
    result = tmp_path / "benchmark.json"
    config = PipelineConfig(
        parse_source(str(video_path)),
        no_display=True,
        benchmark=True,
        warmup=2,
        max_frames=3,
        benchmark_output=result,
    )
    assert run_pipeline(config) == 3
    document = json.loads(result.read_text())
    assert document["completed_warmup_frames"] == 2
    assert document["completed_measured_frames"] == 3
    assert document["requested_measured_frames"] == 3
    assert document["mode"] == "passthrough"
    assert all(
        document["timings"][name]["mean_ms"] == 0
        for name in ("preprocess", "inference", "postprocess")
    )
    assert document["timings"]["capture"]["count"] == 3
    assert document["effective_fps"] is not None
    assert result.with_suffix(".csv").is_file()


def test_benchmark_eof_before_measurement_fails_and_preserves_results(
    video_path: Path, tmp_path: Path
) -> None:
    result = tmp_path / "existing.json"
    csv_path = result.with_suffix(".csv")
    result.write_text('{"existing": true}\n')
    csv_path.write_text("existing,csv\n")
    config = PipelineConfig(
        parse_source(str(video_path)),
        no_display=True,
        benchmark=True,
        warmup=6,
        benchmark_output=result,
    )
    with pytest.raises(BenchmarkError, match="before one measured frame"):
        run_pipeline(config)
    assert result.read_text() == '{"existing": true}\n'
    assert csv_path.read_text() == "existing,csv\n"


def test_detector_benchmark_reuses_session_and_reports_metadata(
    video_path: Path, model_path: Path, tmp_path: Path, mocked_session
) -> None:
    session, constructor = mocked_session
    result = tmp_path / "detector.json"
    config = PipelineConfig(
        parse_source(str(video_path)),
        no_display=True,
        benchmark=True,
        warmup=1,
        max_frames=2,
        benchmark_output=result,
        model=model_path,
    )
    assert run_pipeline(config) == 2
    document = json.loads(result.read_text())
    constructor.assert_called_once()
    assert session.run.call_count == 3
    assert document["mode"] == "detector"
    assert document["provider"] == "CPUExecutionProvider"
    assert document["model_sha256"]
    assert document["completed_warmup_frames"] == 1
    assert document["completed_measured_frames"] == 2
    assert all(document["timings"][name]["count"] == 2 for name in document["timings"])


def test_benchmark_interrupt_does_not_publish_result(
    video_path: Path, tmp_path: Path, monkeypatch
) -> None:
    result = tmp_path / "result.json"
    original = Renderer.render_processing
    calls = 0

    def interrupted(self, frame, detections=()):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(self, frame, detections)

    monkeypatch.setattr(Renderer, "render_processing", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            PipelineConfig(
                parse_source(str(video_path)),
                no_display=True,
                benchmark=True,
                warmup=0,
                benchmark_output=result,
            )
        )
    assert not result.exists()
    assert not result.with_suffix(".csv").exists()
