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


def test_detector_config_default_labels(model_path: Path) -> None:
    config = PipelineConfig(parse_source("0"), model=model_path)
    assert len(config.label_names) == 80
    assert config.label_names[0] == "person"
    assert config.label_names[-1] == "toothbrush"


def test_custom_labels(model_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    names = tuple(f"name {index}" for index in range(80))
    path.write_text("\n".join(names) + "\n")
    assert PipelineConfig(parse_source("0"), model=model_path, labels=path).label_names == names


@pytest.mark.parametrize(
    "contents", [b"", b"one\n", b"one\n" * 80, b"\n" + b"one\n" * 79, b"\xff\xfe"]
)
def test_invalid_labels(model_path: Path, tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "labels.txt"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="[Ll]abels"):
        PipelineConfig(parse_source("0"), model=model_path, labels=path)


def test_labels_require_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --model"):
        PipelineConfig(parse_source("0"), labels=tmp_path / "labels.txt")


@pytest.mark.parametrize("option", ["model", "labels"])
def test_missing_detector_paths(model_path: Path, tmp_path: Path, option: str) -> None:
    options = {"model": model_path, option: tmp_path / "missing"}
    with pytest.raises(ValueError, match="missing or not a regular file"):
        PipelineConfig(parse_source("0"), **options)


def test_unreadable_model(model_path: Path, monkeypatch) -> None:
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if path == model_path:
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ValueError, match="Cannot read --model"):
        PipelineConfig(parse_source("0"), model=model_path)


@pytest.mark.parametrize("option", ["--model", "--labels"])
def test_empty_detector_cli_paths(option: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", option, ""])
    assert error.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "1.5", "bad", ""])
def test_invalid_warmup(value: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", "--warmup", value])
    assert error.value.code == 2
    assert "--warmup" in capsys.readouterr().err


def test_benchmark_requires_headless_mode(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", "--benchmark"])
    assert error.value.code == 2
    assert "--benchmark requires --no-display" in capsys.readouterr().err


def test_benchmark_output_requires_mode(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", "--benchmark-output", "result.json"])
    assert error.value.code == 2
    assert "requires --benchmark" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["result.csv", "result", ""])
def test_benchmark_output_must_be_json(value: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", "0", "--benchmark", "--no-display", "--benchmark-output", value])
    assert error.value.code == 2
    assert "benchmark-output" in capsys.readouterr().err
