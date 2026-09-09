#!/usr/bin/env python3
"""Download the pinned official YOLOX-Nano asset; no export framework required."""

import argparse
import http.client
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

# Allow use before installing the package; model_contract imports only the stdlib.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))
from vision_pipeline.model_contract import (
    MODEL_BYTES,
    MODEL_SHA256,
    MODEL_URL,
    DetectorError,
    verify_model_file,
)


def acquire_model(destination: Path, *, verify_only: bool = False) -> None:
    temporary: Path | None = None
    try:
        if verify_only:
            verify_model_file(destination)
            return
        if destination.exists():
            if not destination.is_file():
                raise DetectorError(
                    f"Model destination is not a regular file: {destination}. "
                    "Choose a file path with --output."
                )
            try:
                verify_model_file(destination)
                return
            except DetectorError:
                # Repair only after a replacement has independently passed verification.
                pass
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            MODEL_URL,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "vision-pipeline-model-helper/0.1",
            },
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}-",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream,
        ):
            temporary = Path(stream.name)
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MODEL_BYTES:
                    raise DetectorError(
                        "Download exceeds the audited model size. "
                        "The upstream response has changed; destination was not replaced."
                    )
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        verify_model_file(temporary)
        temporary.replace(destination)
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        ConnectionError,
    ) as exc:
        raise DetectorError(
            f"Cannot download {MODEL_URL}: {exc}. Check network, proxy/TLS settings, "
            "and GitHub availability/rate limits, then retry."
        ) from exc
    except OSError as exc:
        raise DetectorError(
            f"Cannot publish model to {destination}: {exc}. "
            "Check the destination, permissions, and free space."
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise DetectorError(
                    f"Cannot remove temporary download {temporary}: {exc}. Remove it manually."
                ) from exc


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "detector.onnx"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify local bytes without network or writes",
    )
    args = parser.parse_args(arguments)
    try:
        acquire_model(args.output.expanduser(), verify_only=args.verify_only)
    except DetectorError as exc:
        print(f"model helper: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Model download interrupted; destination was not replaced.", file=sys.stderr
        )
        return 130
    print(f"Verified {args.output}: {MODEL_BYTES} bytes, SHA-256 {MODEL_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
