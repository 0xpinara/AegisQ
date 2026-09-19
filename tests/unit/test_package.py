"""Phase 0: the package imports, reports a version and the CLI runs."""

from __future__ import annotations

import subprocess
import sys

import aegisq
from aegisq.cli.main import main
from aegisq.runtime import hardware


def test_version_is_exposed():
    assert aegisq.__version__ == "0.1.0"


def test_package_imports_without_native_core():
    # native_core() must return None rather than raising on an unbuilt tree.
    core = aegisq.native_core()
    assert core is None or hasattr(core, "build_info")


def test_doctor_runs_and_reports_components(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "AegisQ-HPC" in out
    for expected in ("Python", "CPU", "MPI", "liboqs"):
        assert expected in out


def test_doctor_json_snapshot_is_serialisable(capsys):
    import json

    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "components" in payload
    assert payload["python"].startswith("3.")


def test_environment_snapshot_has_no_exceptions():
    snapshot = hardware.environment_snapshot()
    assert set(snapshot) >= {"platform", "system", "machine", "python", "components"}


def test_cli_version_flag():
    proc = subprocess.run(
        [sys.executable, "-m", "aegisq.cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "AegisQ-HPC 0.1.0" in proc.stdout
