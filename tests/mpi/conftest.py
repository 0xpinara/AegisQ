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


def run_distributed(circuit, mapping=None, precision: str = "fp64"):
    """Execute a circuit on the distributed runtime and gather the full state.

    Gathering defeats the purpose of distributing the state, which is exactly
    why it is confined to the test suite: it is the only way to compare a
    sharded result against the single-process reference amplitude by
    amplitude.
    """
    from aegisq.runtime.native import to_native_circuit

    state = distributed.new_distributed_state(
        circuit.num_qubits, precision=precision, mapping=mapping
    )
    state.apply_circuit(to_native_circuit(circuit))
    return state


def assert_matches_reference(circuit, mapping=None, precision: str = "fp64", atol: float = 1e-11):
    """The distributed result must equal the reference state exactly."""
    import numpy as np

    from aegisq.runtime import Simulator

    state = run_distributed(circuit, mapping=mapping, precision=precision)
    gathered = state.gather()
    expected = Simulator("reference").run(circuit).statevector

    # fp32 shards accumulate visibly more rounding error than fp64, so the
    # norm tolerance follows the amplitude tolerance rather than being fixed.
    assert state.norm() == pytest.approx(1.0, abs=max(atol, 1e-10))
    assert np.allclose(gathered, expected, atol=atol), (
        f"distributed result differs from the reference for {circuit.name}"
    )
    return state


@pytest.fixture(scope="session")
def mpi_rank() -> int:
    if not distributed.mpi_compiled():
        pytest.skip("native core built without MPI")
    return distributed.rank()
