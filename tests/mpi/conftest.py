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


#: Set by every MPI launcher we support. Their presence means somebody
#: deliberately started this process through `mpirun`.
_LAUNCHER_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS", "SLURM_NTASKS")


def _launched_by_mpi() -> bool:
    import os

    return any(name in os.environ for name in _LAUNCHER_VARS)


def _require_mpi_core() -> None:
    """Skip on a plain build; fail when MPI was actually asked for.

    Skipping is the honest outcome when someone runs `pytest tests/mpi`
    on a core built without MPI -- there is nothing to test. Under a
    launcher it is the opposite of honest: the whole suite skips, the
    script exits zero, and a build with no distributed support at all
    reports the same result as one that passed every test. That happened
    here with an interpreter whose version did not match the compiled
    extension, and 166 silent skips looked exactly like success.
    """
    if distributed.mpi_compiled():
        return
    message = "native core built without MPI"
    if _launched_by_mpi():
        pytest.fail(f"{message}, but this process was launched by an MPI launcher")
    pytest.skip(message)


@pytest.fixture(scope="session")
def mpi_world() -> int:
    _require_mpi_core()
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
    _require_mpi_core()
    return distributed.rank()
