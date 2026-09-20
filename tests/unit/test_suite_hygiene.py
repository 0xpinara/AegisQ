"""Checks on the test suite itself.

A test suite is a measuring instrument, and these are its calibration
marks. Both properties below were broken at the same time, and neither
broke a test: one silently imported another project's fixtures, the other
silently ran nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tests_package_resolves_inside_this_repository():
    """`tests` must be this repository's package, not whichever one is first.

    Every module here imports its fixtures as `from tests.conftest import
    ...`. Without `__init__.py`, `tests` is only a namespace *portion*: the
    import machinery notes the directory and keeps scanning `sys.path`, and
    the first real `tests/__init__.py` anywhere on the path wins outright.

    On a machine with an unrelated editable install, that is exactly what
    happened -- `easy-install.pth` put a sibling checkout on the path, its
    `tests` was a regular package, and this suite's imports resolved into
    it. The names happened not to match, so it surfaced as a collection
    error. Had they matched, the suite would have passed against a
    stranger's fixtures.
    """
    import tests

    assert Path(tests.__file__).resolve().parent == REPO_ROOT / "tests"


@pytest.mark.parametrize(
    "package",
    ["tests", "tests.unit", "tests.integration", "tests.mpi", "tests.crypto"],
)
def test_every_test_directory_is_a_real_package(package):
    """One missing `__init__.py` reopens the hole for that directory alone."""
    module = __import__(package, fromlist=["__file__"])
    assert module.__file__ is not None, f"{package} is a namespace package"
    assert Path(module.__file__).name == "__init__.py"


def test_missing_mpi_support_fails_under_a_launcher(monkeypatch):
    """A launcher was used, so "nothing to test" is a failure, not a skip.

    Skipping is right when someone runs `pytest tests/mpi` on a build
    without MPI. Under `mpirun` it means the distributed suite proved
    nothing while exiting zero -- 166 skips that read exactly like 166
    passes.
    """
    from aegisq.runtime import distributed
    from tests.mpi import conftest

    monkeypatch.setattr(distributed, "mpi_compiled", lambda: False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    with pytest.raises(BaseException) as caught:
        conftest._require_mpi_core()
    assert "launched by an MPI launcher" in str(caught.value)


def test_missing_mpi_support_skips_without_a_launcher(monkeypatch):
    """Outside a launcher there is genuinely nothing to run."""
    from aegisq.runtime import distributed
    from tests.mpi import conftest

    monkeypatch.setattr(distributed, "mpi_compiled", lambda: False)
    for name in conftest._LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(BaseException) as caught:
        conftest._require_mpi_core()
    assert type(caught.value).__name__ == "Skipped", f"expected a skip, got {caught.value!r}"
