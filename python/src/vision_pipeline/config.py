"""Configuration validation without importing OpenCV or opening media devices."""

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from vision_pipeline.model_contract import CLASS_COUNT, model_context

DEFAULT_LABELS = Path(__file__).resolve().parents[3] / "models" / "classes.txt"

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm"}
)
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
)
# Container/codec pairs supported for writing; availability depends on OpenCV's backend.
OUTPUT_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".mov": "mp4v", ".mkv": "MJPG", ".webm": "VP80"}


class SourceKind(Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"


@dataclass(frozen=True)
class SourceSpec:
    kind: SourceKind
    location: Path | int


def parse_source(value: str) -> SourceSpec:
    """Bare ASCII digits select a camera; files require a supported extension."""
    if not value or not value.strip():
        raise ValueError(
            "--source must be a non-negative camera index or an image/video file path."
        )
    if re.fullmatch(r"[0-9]+", value):
        index = int(value)
        if index > 2**31 - 1:
            raise ValueError("Camera index is too large; use an index between 0 and 2147483647.")
        return SourceSpec(SourceKind.CAMERA, index)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", value):
        raise ValueError("Camera index must be a non-negative integer, such as --source 0.")
    if "://" in value:
        raise ValueError("Unsupported source URL; use a local image/video file or a camera index.")
    path = Path(value).expanduser()
    try:
        if not path.exists():
            raise ValueError(f"Source file does not exist: {path}. Check the --source path.")
        if not path.is_file():
            raise ValueError(
                f"Source is not a regular file: {path}. Select an image or video file."
            )
    except OSError as exc:
        raise ValueError(f"Cannot access source {path}: {exc}") from exc
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return SourceSpec(SourceKind.IMAGE, path)
    if suffix in VIDEO_EXTENSIONS:
        return SourceSpec(SourceKind.VIDEO, path)
    raise ValueError(
        f"Unsupported source extension {suffix or '(none)'}: {path}. "
        "Use an image such as PNG/JPEG or a video such as AVI/MP4."
    )


def unit_interval(value: str | float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1.") from exc
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")
    return result


def positive_integer(value: str) -> int:
    if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
        raise ValueError("--max-frames must be a positive integer.")
    return int(value)


def validate_readable_file(path: Path, option: str) -> None:
    try:
        if not path.is_file():
            raise ValueError(f"{option} file is missing or not a regular file: {path}.")
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise ValueError(f"Cannot read {option} file {path}: {exc}. Check permissions.") from exc


def load_labels(path: Path) -> tuple[str, ...]:
    validate_readable_file(path, "--labels")
    try:
        labels = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Cannot read labels {path}: {exc}. Use a readable UTF-8 class-name file."
        ) from exc
    if (
        len(labels) != CLASS_COUNT
        or any(not label for label in labels)
        or len(set(labels)) != CLASS_COUNT
    ):
        raise ValueError(
            f"Labels {path} must contain exactly {CLASS_COUNT} nonempty, unique names, "
            "one per line in COCO order. Use models/classes.txt."
        )
    return labels


@dataclass(frozen=True)
class PipelineConfig:
    source: SourceSpec
    output: Path | None = None
    no_display: bool = False
    max_frames: int | None = None
    confidence: float = 0.25
    iou: float = 0.45
    model: Path | None = None
    labels: Path | None = None
    label_names: tuple[str, ...] = field(init=False, default=(), repr=False)

    def __post_init__(self) -> None:
        unit_interval(self.confidence, "--confidence")
        unit_interval(self.iou, "--iou")
        if self.max_frames is not None and (
            type(self.max_frames) is not int or self.max_frames < 1
        ):
            raise ValueError("--max-frames must be a positive integer.")
        if self.model is not None:
            try:
                validate_readable_file(self.model, "--model")
            except ValueError as exc:
                raise ValueError(model_context(self.model, str(exc))) from exc
            if self.labels is None:
                object.__setattr__(self, "labels", DEFAULT_LABELS)
            object.__setattr__(self, "label_names", load_labels(self.labels))
        elif self.labels is not None:
            raise ValueError("--labels requires --model; omit both for media passthrough.")
        if self.output is not None:
            self._validate_output()

    def _validate_output(self) -> None:
        assert self.output is not None
        path = self.output
        extensions = IMAGE_EXTENSIONS if self.source.kind is SourceKind.IMAGE else OUTPUT_CODECS
        if path.suffix.lower() not in extensions:
            raise ValueError(
                f"Unsupported output extension for {self.source.kind.value}: {path}. "
                f"Choose one of: {', '.join(sorted(extensions))}."
            )
        try:
            if path.exists() and not path.is_file():
                raise ValueError(
                    f"Output is not a regular file: {path}. Choose an output filename."
                )
            if isinstance(self.source.location, Path):
                source = self.source.location
                if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
                    raise ValueError(
                        "Output must differ from the source file; choose another path."
                    )
            for name, protected in (("model", self.model), ("labels", self.labels)):
                if protected is not None and (
                    path.resolve() == protected.resolve()
                    or (path.exists() and path.samefile(protected))
                ):
                    raise ValueError(
                        f"Output must differ from the {name} file; choose another path."
                    )
            for parent in path.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"Output parent is not a directory: {parent}.")
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Invalid output path {path}: {exc}") from exc
