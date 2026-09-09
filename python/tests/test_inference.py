from pathlib import Path

import numpy as np
import pytest

from vision_pipeline.inference import InferenceEngine
from vision_pipeline.model_contract import INPUT_SHAPE, OUTPUT_SHAPE, PROVIDER, DetectorError


def test_names_cpu_session_reuse_and_close(model_path: Path, mocked_session) -> None:
    session, constructor = mocked_session
    tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)
    with InferenceEngine(model_path) as engine:
        assert engine.run(tensor) is session.run.return_value[0]
        engine.run(tensor)
    constructor.assert_called_once_with(str(model_path), providers=[PROVIDER])
    assert session.run.call_count == 2
    args = session.run.call_args.args
    assert args[0] == ["actual_output"]
    assert args[1]["actual_input"] is tensor
    with pytest.raises(DetectorError, match="Session is closed"):
        engine.run(tensor)


@pytest.mark.parametrize(
    "target, field, value",
    [
        ("input", "shape", [1, 3, 640, 640]),
        ("input", "shape", ["batch", 3, 416, 416]),
        ("input", "type", "tensor(float16)"),
        ("output", "shape", [1, 3549, 84]),
        ("output", "shape", [1, 85, 3549]),
        ("output", "type", "tensor(double)"),
    ],
)
def test_incompatible_metadata_is_actionable(
    model_path: Path, mocked_session, target, field, value
) -> None:
    session, _ = mocked_session
    items = (
        session.get_inputs.return_value if target == "input" else session.get_outputs.return_value
    )
    setattr(items[0], field, value)
    with pytest.raises(DetectorError) as error:
        InferenceEngine(model_path)
    message = str(error.value)
    for text in (
        str(model_path),
        PROVIDER,
        "observed",
        "expected",
        "actual_input",
        "actual_output",
        str(value),
    ):
        assert text in message


@pytest.mark.parametrize("target", ["inputs", "outputs", "providers"])
def test_invalid_counts_and_provider(model_path: Path, mocked_session, target) -> None:
    session, _ = mocked_session
    getattr(session, f"get_{target}").return_value = []
    with pytest.raises(DetectorError, match="Incompatible model"):
        InferenceEngine(model_path)


def test_native_creation_failure_has_context(model_path: Path, mocked_session) -> None:
    _, constructor = mocked_session
    constructor.side_effect = RuntimeError("native load failure")
    with pytest.raises(DetectorError) as error:
        InferenceEngine(model_path)
    assert all(
        part in str(error.value)
        for part in (str(model_path), PROVIDER, "expected", "unavailable", "native load failure")
    )


def test_native_run_failure_has_context(model_path: Path, mocked_session) -> None:
    session, _ = mocked_session
    session.run.side_effect = RuntimeError("native run failure")
    with (
        InferenceEngine(model_path) as engine,
        pytest.raises(DetectorError, match="native run failure") as error,
    ):
        engine.run(np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert "actual_output" in str(error.value)
    assert str(model_path) in str(error.value)


@pytest.mark.parametrize(
    "output",
    [
        [],
        [np.zeros((1, 1, 85), dtype=np.float32)],
        [np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32)],
    ],
)
def test_runtime_output_validation(model_path: Path, mocked_session, output) -> None:
    session, _ = mocked_session
    session.run.return_value = output
    with InferenceEngine(model_path) as engine, pytest.raises(DetectorError):
        engine.run(np.zeros(INPUT_SHAPE, dtype=np.float32))


@pytest.mark.parametrize(
    "tensor",
    [
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        np.zeros(INPUT_SHAPE, dtype=np.float64),
        np.zeros(INPUT_SHAPE, dtype=np.float32)[:, :, :, ::-1],
        np.full(INPUT_SHAPE, np.inf, dtype=np.float32),
    ],
)
def test_invalid_input_never_reaches_session(model_path: Path, mocked_session, tensor) -> None:
    session, _ = mocked_session
    with (
        InferenceEngine(model_path) as engine,
        pytest.raises(DetectorError, match="contiguous finite"),
    ):
        engine.run(tensor)
    session.run.assert_not_called()


def test_corrupt_model_rejected_without_session_creation(model_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Corrupt bytes must fail before session construction")

    monkeypatch.setattr("vision_pipeline.inference.ort.InferenceSession", forbidden)
    with pytest.raises(DetectorError, match="checksum/size mismatch"):
        InferenceEngine(model_path)


def test_missing_model_in_engine(tmp_path: Path) -> None:
    with pytest.raises(DetectorError, match="missing or not a regular file") as error:
        InferenceEngine(tmp_path / "missing.onnx")
    assert PROVIDER in str(error.value)
