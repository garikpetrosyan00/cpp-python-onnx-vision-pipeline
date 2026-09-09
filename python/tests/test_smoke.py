"""Explicit local-model smoke; never downloads and skips only if the model is absent."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipeline.config import DEFAULT_LABELS, load_labels
from vision_pipeline.inference import InferenceEngine
from vision_pipeline.model_contract import verify_model_file
from vision_pipeline.postprocess import postprocess
from vision_pipeline.preprocess import preprocess


@pytest.mark.real_model
def test_real_model_headless_image(tmp_path: Path) -> None:
    model = Path(__file__).parents[2] / "models" / "detector.onnx"
    if not model.exists():
        pytest.skip(
            "Local model absent; run scripts/download_or_export_model.py, then this test explicitly"
        )
    verify_model_file(model)  # A present but corrupt model must fail, not skip.
    y, x = np.indices((240, 320))
    frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    source = tmp_path / "generated.png"
    output = tmp_path / "result" / "annotated.png"
    assert cv2.imwrite(str(source), frame)
    prepared = preprocess(frame)
    with InferenceEngine(model) as engine:
        detections = postprocess(
            engine.run(prepared.tensor), prepared.metadata, load_labels(DEFAULT_LABELS), 0.01
        )
    # Intentionally low threshold exercises drawing on synthetic data, not detection accuracy.
    assert detections
    command = [
        sys.executable,
        "-m",
        "vision_pipeline",
        "--model",
        str(model),
        "--labels",
        str(DEFAULT_LABELS),
        "--source",
        str(source),
        "--output",
        str(output),
        "--no-display",
        "--max-frames",
        "1",
        "--confidence",
        "0.01",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr
    saved = cv2.imread(str(output))
    assert saved is not None and saved.shape == frame.shape and saved.dtype == frame.dtype
    assert np.any(saved != frame)
    np.testing.assert_array_equal(cv2.imread(str(source)), frame)
    assert list(output.parent.iterdir()) == [output]
