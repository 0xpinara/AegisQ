"""Shared pytest fixtures and availability guards for the AegisQ test suite."""

from __future__ import annotations

import pytest

import aegisq

#: Tolerance used when comparing amplitudes produced by different backends.
#: fp64 state-vector arithmetic accumulates error proportional to circuit
#: depth; 1e-10 is loose enough for the depths used in tests and tight enough
#: to catch a genuine kernel bug.
AMPLITUDE_TOL = 1e-10


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "native: requires the compiled C++ core")


@pytest.fixture(scope="session")
def tol() -> float:
    return AMPLITUDE_TOL


@pytest.fixture(scope="session")
def native_core():
    core = aegisq.native_core()
    if core is None:
        pytest.skip("native core is not built (run `make build`)")
    return core


def requires_qiskit():
    return pytest.mark.skipif(
        not _importable("qiskit"), reason="Qiskit is not installed (validation extra)"
    )


def requires_liboqs():
    return pytest.mark.skipif(
        not _importable("oqs"), reason="liboqs-python is not installed (crypto extra)"
    )


def _importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
