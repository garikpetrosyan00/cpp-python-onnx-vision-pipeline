"""One context-managed iterator for images, video files, and cameras."""

import math
from collections.abc import Iterator
from types import TracebackType
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import SourceKind, SourceSpec

Frame = NDArray[np.uint8]


class MediaError(RuntimeError):
    """An actionable acquisition, output, or display failure."""


class InputSource(Iterator[Frame]):
    def __init__(self, source: SourceSpec) -> None:
        self.source = source
        self.fps = 30.0  # Fallback when a camera/backend supplies no usable frame rate.
        self._capture: cv2.VideoCapture | None = None
        self._image: Frame | None = None
        self._opened = False
        self._frames = 0
        self._expected_frames = 0
        self._eof = False

    def __enter__(self) -> Self:
        if self._opened:
            raise MediaError("Input source is already open; use one context at a time.")
        self._frames = 0
        self._expected_frames = 0
        self._eof = False
        self.fps = 30.0
        try:
            if self.source.kind is SourceKind.IMAGE:
                self._image = cv2.imread(str(self.source.location), cv2.IMREAD_COLOR)
                if self._image is None or self._image.size == 0:
                    raise MediaError(
                        f"Cannot read image {self.source.location}. "
                        "Check file permissions and that it is a valid supported image."
                    )
            else:
                self._capture = cv2.VideoCapture()
                location = self.source.location
                self._capture.open(location if isinstance(location, int) else str(location))
                if not self._capture.isOpened():
                    if self.source.kind is SourceKind.CAMERA:
                        raise MediaError(
                            f"Camera {location} is unavailable. Check the index, "
                            "device permissions, connection, and whether another app is using it."
                        )
                    raise MediaError(
                        f"Cannot open video {location}. Check permissions and codec support; "
                        "the file may be empty or damaged."
                    )
                fps = self._capture.get(cv2.CAP_PROP_FPS)
                if math.isfinite(fps) and fps > 0:
                    self.fps = fps
                count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if self.source.kind is SourceKind.VIDEO and math.isfinite(count) and count > 0:
                    self._expected_frames = int(count)
            self._opened = True
            return self
        except cv2.error as exc:
            self.close()
            raise MediaError(
                f"Cannot open {self.source.kind.value} {self.source.location}: {exc}. "
                "Check the source, permissions, and OpenCV backend support."
            ) from exc
        except BaseException:
            self.close()
            raise

    def __next__(self) -> Frame:
        if not self._opened:
            raise MediaError("Input source is closed; acquire frames inside its context manager.")
        if self._eof:
            raise StopIteration
        if self.source.kind is SourceKind.IMAGE:
            assert self._image is not None
            self._eof = True
            return self._image
        assert self._capture is not None
        try:
            ok, frame = self._capture.read()
        except cv2.error as exc:
            raise MediaError(
                f"Failed to read {self.source.kind.value} {self.source.location}: {exc}"
            ) from exc
        if not ok or frame is None or frame.size == 0:
            if self.source.kind is SourceKind.CAMERA:
                raise MediaError(
                    f"Cannot read a frame from camera {self.source.location}. "
                    "Check its connection and whether it is in use."
                )
            if self._frames == 0:
                raise MediaError(
                    f"Video {self.source.location} is empty or has no decodable frames. "
                    "Check the file and installed codec support."
                )
            if self._frames < self._expected_frames:
                raise MediaError(
                    f"Video {self.source.location} became unreadable after {self._frames} frames "
                    f"(expected {self._expected_frames}). Check for a damaged or truncated file."
                )
            self._eof = True
            raise StopIteration
        self._frames += 1
        return frame

    def close(self) -> None:
        try:
            if self._capture is not None:
                self._capture.release()
        finally:
            self._capture = None
            self._image = None
            self._opened = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
