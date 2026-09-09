"""No-network tests of the real helper's checksum and publication lifecycle."""

import hashlib
import http.client
import importlib.util
import io
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vision_pipeline import model_contract
from vision_pipeline.model_contract import DetectorError


@pytest.fixture
def helper(monkeypatch):
    path = Path(__file__).parents[2] / "scripts" / "download_or_export_model.py"
    spec = importlib.util.spec_from_file_location("model_download_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"deterministic verified model fixture"
    digest = hashlib.sha256(payload).hexdigest()
    # Small known bytes exercise the same verification function used for the real model.
    for target in (module, model_contract):
        monkeypatch.setattr(target, "MODEL_BYTES", len(payload))
        monkeypatch.setattr(target, "MODEL_SHA256", digest)
    network = MagicMock(side_effect=lambda *args, **kwargs: io.BytesIO(payload))
    monkeypatch.setattr(module.urllib.request, "urlopen", network)
    return module, payload, network


def test_download_verify_atomic_publish_and_idempotence(
    helper, tmp_path: Path, monkeypatch
) -> None:
    module, payload, network = helper
    output = tmp_path / "nested" / "detector.onnx"
    output.parent.mkdir()
    output.write_bytes(b"old corrupt file")
    original_replace = Path.replace

    def checked_replace(temporary, destination):
        assert temporary.read_bytes() == payload
        assert destination.read_bytes() == b"old corrupt file"
        return original_replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", checked_replace)
    module.acquire_model(output)
    module.acquire_model(output)
    module.acquire_model(output, verify_only=True)
    assert output.read_bytes() == payload
    network.assert_called_once()
    request = network.call_args.args[0]
    assert request.full_url == model_contract.MODEL_URL
    assert request.get_header("Accept") == "application/octet-stream"
    assert list(output.parent.iterdir()) == [output]


def test_verify_only_never_downloads_or_writes(helper, tmp_path: Path) -> None:
    module, _, network = helper
    output = tmp_path / "missing" / "detector.onnx"
    with pytest.raises(DetectorError, match="missing"):
        module.acquire_model(output, verify_only=True)
    assert not output.parent.exists()
    network.assert_not_called()


@pytest.mark.parametrize(
    "kind", ["checksum", "oversized", "truncated", "network", "partial", "interrupt"]
)
def test_failure_preserves_destination_and_cleans_temp(helper, tmp_path: Path, kind: str) -> None:
    module, payload, network = helper
    output = tmp_path / "detector.onnx"
    output.write_bytes(b"existing data")
    expected = DetectorError
    if kind == "network":
        network.side_effect = urllib.error.URLError("connection unavailable")
    elif kind in ("partial", "interrupt"):
        response = MagicMock()
        response.__enter__.return_value = response
        failure = (
            http.client.IncompleteRead(b"incomplete") if kind == "partial" else KeyboardInterrupt()
        )
        response.read.side_effect = [payload[:5], failure]
        network.side_effect = lambda *args, **kwargs: response
        expected = DetectorError if kind == "partial" else KeyboardInterrupt
    else:
        body = {
            "checksum": b"x" * len(payload),
            "oversized": payload + b"x",
            "truncated": payload[:5],
        }[kind]
        network.side_effect = lambda *args, **kwargs: io.BytesIO(body)
    with pytest.raises(expected):
        module.acquire_model(output)
    assert output.read_bytes() == b"existing data"
    assert list(tmp_path.iterdir()) == [output]


def test_filesystem_error_is_actionable(helper, tmp_path: Path, monkeypatch) -> None:
    module, _, _ = helper
    denied = MagicMock(side_effect=PermissionError("denied"))
    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", denied)
    with pytest.raises(DetectorError, match="permissions, and free space"):
        module.acquire_model(tmp_path / "detector.onnx")
    assert list(tmp_path.iterdir()) == []


def test_helper_cli_error_and_ctrl_c(helper, tmp_path: Path, capsys) -> None:
    module, _, network = helper
    output = tmp_path / "detector.onnx"
    assert module.main(["--output", str(output), "--verify-only"]) == 1
    assert "missing" in capsys.readouterr().err
    network.side_effect = KeyboardInterrupt()
    assert module.main(["--output", str(output)]) == 130
    assert "interrupted" in capsys.readouterr().err
    assert not output.exists()


def test_destination_access_error_is_actionable(helper, tmp_path: Path, monkeypatch) -> None:
    module, _, network = helper
    monkeypatch.setattr(Path, "exists", MagicMock(side_effect=PermissionError("parent denied")))
    with pytest.raises(DetectorError, match="permissions, and free space"):
        module.acquire_model(tmp_path / "detector.onnx")
    network.assert_not_called()
