"""Fixtures for tests that must be launched through mpirun.

Every rank executes the same test bodies. Assertions are written so that they
hold on all ranks; rank-specific expectations are derived from the fixtures
below rather than hard-coded.
"""

from __future__ import annotations

import pytest

from aegisq.runtime import distributed


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mpi: launched through scripts/run_mpi_tests.sh")


@pytest.fixture(scope="session")
def mpi_world() -> int:
    if not distributed.mpi_compiled():
        pytest.skip("native core built without MPI")
    return distributed.world_size()


@pytest.fixture(scope="session")
def mpi_rank() -> int:
    if not distributed.mpi_compiled():
        pytest.skip("native core built without MPI")
    return distributed.rank()
