import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vision_pipeline import __version__
from vision_pipeline.config import PipelineConfig, parse_source, positive_integer, unit_interval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-pipeline",
        description="Python media pipeline (Phase 1 passthrough; no AI inference)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--source", help="Non-negative camera index (e.g. 0), image, or video path")
    parser.add_argument("--output", help="Save to an image path or video path (e.g. .avi or .mp4)")
    parser.add_argument("--no-display", action="store_true", help="Run without GUI windows")
    parser.add_argument("--max-frames", help="Stop after this positive number of frames")
    parser.add_argument("--confidence", default="0.25", help="Number in [0, 1]; unused in Phase 1")
    parser.add_argument("--iou", default="0.45", help="Number in [0, 1]; unused in Phase 1")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(arguments) if arguments is not None else sys.argv[1:]
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.source is None:
        parser.error("--source is required to run the media pipeline.")
    try:
        config = PipelineConfig(
            source=parse_source(args.source),
            output=Path(args.output).expanduser() if args.output else None,
            no_display=args.no_display,
            max_frames=positive_integer(args.max_frames) if args.max_frames is not None else None,
            confidence=unit_interval(args.confidence, "--confidence"),
            iou=unit_interval(args.iou, "--iou"),
        )
        if args.output == "":
            raise ValueError("--output must be a non-empty image or video path.")
    except ValueError as exc:
        parser.error(str(exc))

    # Keep package import, help, and version usable without runtime dependencies.
    from vision_pipeline.input_source import MediaError
    from vision_pipeline.pipeline import run_pipeline

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("Interrupted; media resources released.", file=sys.stderr)
        return 130
    except MediaError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1
    return 0
