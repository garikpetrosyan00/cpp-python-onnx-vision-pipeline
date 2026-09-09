from vision_pipeline import __version__
from vision_pipeline.cli import build_parser, main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_parser_identity() -> None:
    assert build_parser().prog == "vision-pipeline"


def test_main_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "Phase 2 CPU detector" in capsys.readouterr().out


def test_help_and_version_without_runtime_dependencies() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    for option, expected in [("--help", "--source"), ("--version", __version__)]:
        result = subprocess.run(
            [sys.executable, "-S", "-m", "vision_pipeline", option],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
