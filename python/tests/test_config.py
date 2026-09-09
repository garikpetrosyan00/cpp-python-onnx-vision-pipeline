from pathlib import Path

import pytest

from vision_pipeline.cli import main
from vision_pipeline.config import (
    PipelineConfig,
    SourceKind,
    parse_source,
    positive_integer,
    unit_interval,
)


@pytest.mark.parametrize("value, expected", [("0", 0), ("12", 12), ("001", 1)])
def test_camera_index(value: str, expected: int) -> None:
    source = parse_source(value)
    assert source.kind is SourceKind.CAMERA
    assert source.location == expected


@pytest.mark.parametrize("value", ["", " ", "-1", "+1", "1.5", "2147483648"])
def test_invalid_camera_index(value: str) -> None:
    with pytest.raises(ValueError, match="camera|Camera"):
        parse_source(value)


@pytest.mark.parametrize("suffix, kind", [(".PNG", SourceKind.IMAGE), (".mp4", SourceKind.VIDEO)])
def test_file_source(tmp_path: Path, suffix: str, kind: SourceKind) -> None:
    path = tmp_path / f"source{suffix}"
    path.touch()  # Parsing checks paths; decoding is acquisition's responsibility.
    assert parse_source(str(path)).kind is kind
    assert parse_source(str(path)).location == path


def test_source_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        parse_source(str(tmp_path / "missing.png"))
    with pytest.raises(ValueError, match="not a regular file"):
        parse_source(str(tmp_path))
    path = tmp_path / "data.txt"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported source extension"):
        parse_source(str(path))
    with pytest.raises(ValueError, match="Unsupported source URL"):
        parse_source("https://example.test/video.mp4")


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-inf", "-0.01", "1.01", "", True])
def test_invalid_threshold(value) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        unit_interval(value, "--confidence")


@pytest.mark.parametrize("value", ["0", "1", "0.5", "1e-2"])
def test_valid_threshold(value: str) -> None:
    assert unit_interval(value, "--iou") == float(value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", "", "inf"])
def test_invalid_frame_limit(value: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value)


def test_frame_limit() -> None:
    assert positive_integer("10") == 10


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_dataclass_validates_frame_limit(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PipelineConfig(parse_source("0"), max_frames=value)


@pytest.mark.parametrize("field", ["confidence", "iou"])
def test_dataclass_validates_threshold(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        PipelineConfig(parse_source("0"), **{field: float("nan")})


def test_output_validation(image_path: Path, tmp_path: Path) -> None:
    source = parse_source(str(image_path))
    with pytest.raises(ValueError, match="differ from the source"):
        PipelineConfig(source, output=image_path)
    alias = tmp_path / "alias.png"
    alias.symlink_to(image_path)
    with pytest.raises(ValueError, match="differ from the source"):
        PipelineConfig(source, output=alias)
    alias.unlink()
    alias.hardlink_to(image_path)
    with pytest.raises(ValueError, match="differ from the source"):
        PipelineConfig(source, output=alias)
    with pytest.raises(ValueError, match="Unsupported output extension"):
        PipelineConfig(source, output=tmp_path / "out.mp4")
    with pytest.raises(ValueError, match="Unsupported output extension"):
        PipelineConfig(parse_source("0"), output=tmp_path / "out.png")
    directory = tmp_path / "directory.png"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        PipelineConfig(source, output=directory)
    with pytest.raises(ValueError, match="parent is not a directory"):
        PipelineConfig(source, output=image_path / "out.png")
    output = tmp_path / "new" / "out.png"
    assert PipelineConfig(source, output=output).output == output
    assert not output.parent.exists()  # Validation has no write side effects.


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--no-display"], "--source is required"),
        (["--source", "0", "--confidence", "bad"], "--confidence"),
        (["--source", "0", "--iou", "nan"], "--iou"),
        (["--source", "0", "--max-frames", "0"], "--max-frames"),
        (["--source", "0", "--output", ""], "--output must be a non-empty"),
    ],
)
def test_cli_configuration_errors(arguments, message: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
