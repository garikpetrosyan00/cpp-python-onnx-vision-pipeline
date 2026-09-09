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
