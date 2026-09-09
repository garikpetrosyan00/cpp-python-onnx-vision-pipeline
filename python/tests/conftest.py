"""Small deterministic media generated locally; no external assets or devices."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    y, x = np.indices((48, 64))
    frame = np.stack((x * 3, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    path = tmp_path / "input.png"
    assert cv2.imwrite(str(path), frame)
    return path


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    try:
        if not writer.isOpened():
            pytest.skip("OpenCV MJPG/AVI encoder is unavailable")
        for index in range(6):
            writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Readable placeholder only; focused session tests explicitly mock verification."""
    path = tmp_path / "mock.onnx"
    path.write_bytes(b"mock session fixture")
    return path


@pytest.fixture
def mocked_session(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vision_pipeline import inference
    from vision_pipeline.model_contract import INPUT_SHAPE, OUTPUT_SHAPE, PROVIDER

    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="actual_input", shape=list(INPUT_SHAPE), type="tensor(float)")
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="actual_output", shape=list(OUTPUT_SHAPE), type="tensor(float)")
    ]
    session.get_providers.return_value = [PROVIDER]
    session.run.return_value = [np.zeros(OUTPUT_SHAPE, dtype=np.float32)]
    constructor = MagicMock(return_value=session)
    monkeypatch.setattr(inference.ort, "InferenceSession", constructor)
    monkeypatch.setattr(inference, "verify_model_file", lambda path: None)
    return session, constructor
