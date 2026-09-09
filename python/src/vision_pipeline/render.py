"""Annotate detections, display frames, and publish verified media on clean exit."""

import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2

from vision_pipeline.config import OUTPUT_CODECS, PipelineConfig, SourceKind
from vision_pipeline.input_source import Frame, MediaError
from vision_pipeline.postprocess import Detection


def class_color(class_id: int) -> tuple[int, int, int]:
    """Stable BGR colors independent of process hash/random state."""
    return tuple(
        64 + (class_id * multiplier + offset) % 192
        for multiplier, offset in ((37, 29), (67, 83), (97, 137))
    )


def annotate(frame: Frame, detections: Sequence[Detection]) -> Frame:
    """Draw on a copy when needed; retain ownership of the untouched input frame."""
    if not detections:
        return frame
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    try:
        for detection in detections:
            x1 = max(0, min(width - 1, int(detection.x1)))
            y1 = max(0, min(height - 1, int(detection.y1)))
            x2 = max(0, min(width - 1, int(detection.x2)))
            y2 = max(0, min(height - 1, int(detection.y2)))
            color = class_color(detection.class_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.label} {detection.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = max(0, min(x1, width - text_width - 4))
            text_y = min(height - 1, y1 - 5 if y1 > text_height + 5 else y1 + text_height + 2)
            cv2.rectangle(
                canvas,
                (text_x, max(0, text_y - text_height - 2)),
                (min(width - 1, text_x + text_width + 3), min(height - 1, text_y + baseline)),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                text,
                (text_x, max(0, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    except cv2.error as exc:
        raise MediaError(f"Cannot draw detections: {exc}") from exc
    return canvas


class Renderer:
    def __init__(self, config: PipelineConfig, fps: float) -> None:
        self.config = config
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._temporary: Path | None = None
        self._size: tuple[int, int] | None = None
        self._frames = 0
        self._window_open = False
        self._window_name = "Vision pipeline — Q / ESC to exit"

    def __enter__(self) -> Self:
        # Qt builds can abort the process instead of raising when no display exists.
        if (
            not self.config.no_display
            and sys.platform.startswith("linux")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            raise MediaError("No display session found. Run with --no-display for headless use.")
        return self

    def render(self, frame: Frame, detections: Sequence[Detection] = ()) -> bool:
        """Save/display annotated frames; return False when Q or ESC is pressed."""
        frame = annotate(frame, detections)
        if self.config.output is not None:
            self._save(frame)
        if self.config.no_display:
            return True
        try:
            if not self._window_open:
                cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
                self._window_open = True
            cv2.imshow(self._window_name, frame)
            is_image = self.config.source.kind is SourceKind.IMAGE
            delay = 30 if is_image else max(1, min(1000, round(1000 / self.fps)))
            while True:
                if cv2.waitKey(delay) & 0xFF in (ord("q"), ord("Q"), 27):
                    return False
                if not is_image:
                    return True
                # Poll instead of waitKey(0), allowing Python to handle Ctrl+C.
        except cv2.error as exc:
            raise MediaError(
                "Cannot display frames. Use --no-display, or check your GUI session "
                f"and OpenCV GUI support: {exc}"
            ) from exc

    def _save(self, frame: Frame) -> None:
        output = self.config.output
        assert output is not None
        try:
            if self._temporary is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                # Stage in the destination directory so failed runs leave existing output intact.
                with tempfile.NamedTemporaryFile(
                    prefix=f".{output.stem}-", suffix=output.suffix, dir=output.parent, delete=False
                ) as temporary:
                    self._temporary = Path(temporary.name)
            if self.config.source.kind is SourceKind.IMAGE:
                if not cv2.imwrite(str(self._temporary), frame):
                    raise MediaError(
                        f"Cannot write image {output}. Check permissions and encoder support."
                    )
            else:
                height, width = frame.shape[:2]
                size = (width, height)
                if self._writer is None:
                    if width % 2 or height % 2:
                        raise MediaError(
                            f"Cannot save {output}: video output requires even frame dimensions; "
                            f"received {width}x{height}. Select a source with even dimensions "
                            "to avoid codec cropping."
                        )
                    self._size = size
                    codec = OUTPUT_CODECS[output.suffix.lower()]
                    self._writer = cv2.VideoWriter()
                    self._writer.open(
                        str(self._temporary), cv2.VideoWriter_fourcc(*codec), self.fps, size
                    )
                    if not self._writer.isOpened():
                        raise MediaError(
                            f"Cannot open video output {output} with codec {codec}. "
                            "Try .avi (MJPG), or install an OpenCV build with the required encoder."
                        )
                if size != self._size:
                    raise MediaError(
                        f"Cannot save changing frame dimensions to {output}: "
                        f"expected {self._size}, received {size}."
                    )
                self._writer.write(frame)
            self._frames += 1
        except (cv2.error, OSError) as exc:
            raise MediaError(
                f"Cannot write output {output}: {exc}. "
                "Check permissions, free space, and codec support."
            ) from exc

    def _verify_output(self) -> None:
        """Read staged output: OpenCV write() provides no per-frame success status."""
        assert self._temporary is not None
        if self.config.source.kind is SourceKind.IMAGE:
            frame = cv2.imread(str(self._temporary), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise MediaError(
                    f"Saved image is unreadable: {self.config.output}. Check the encoder."
                )
            return
        capture = cv2.VideoCapture()
        try:
            capture.open(str(self._temporary))
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or (frame.shape[1], frame.shape[0]) != self._size:
                    raise MediaError(
                        f"Saved video has invalid frame dimensions: {self.config.output}."
                    )
                count += 1
            if count != self._frames:
                raise MediaError(
                    f"Video output verification failed for {self.config.output}: "
                    f"wrote {self._frames} frames but decoded {count}. "
                    "Check free space and codec support, or try .avi (MJPG)."
                )
        finally:
            capture.release()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            # Q/ESC, max_frames, EOF, and Ctrl+C all keep successfully acquired frames.
            if (
                (exc_type is None or issubclass(exc_type, KeyboardInterrupt))
                and self._temporary is not None
                and self._frames
            ):
                self._verify_output()
                assert self.config.output is not None
                self._temporary.replace(self.config.output)
        except (cv2.error, OSError) as error:
            raise MediaError(
                f"Cannot finalize output {self.config.output}: {error}. "
                "Check permissions, free space, and codec support."
            ) from error
        finally:
            try:
                if self._temporary is not None:
                    self._temporary.unlink(missing_ok=True)
            except OSError as error:
                raise MediaError(
                    f"Cannot remove temporary output {self._temporary}: {error}"
                ) from error
            finally:
                if self._window_open:
                    try:
                        cv2.destroyWindow(self._window_name)
                    except cv2.error as error:
                        if exc_type is None:
                            raise MediaError(f"Cannot close display window: {error}") from error
                    finally:
                        self._window_open = False
