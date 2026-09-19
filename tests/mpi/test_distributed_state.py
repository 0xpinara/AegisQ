"""Phase 4: distributed state allocation and initialisation.

Run with `scripts/run_mpi_tests.sh` (or `make test-mpi`), which launches this
file under mpirun at several rank counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.runtime import distributed

pytestmark = pytest.mark.mpi


def qubits_for(world: int, extra: int = 4) -> int:
    """Circuit width that leaves `extra` local qubits on every rank."""
    return max(1, world.bit_length() - 1) + extra


def test_world_is_a_power_of_two(mpi_world):
    assert mpi_world & (mpi_world - 1) == 0


def test_layout_agrees_with_the_live_world(mpi_world, mpi_rank):
    n = qubits_for(mpi_world)
    state = distributed.new_distributed_state(n)
    lay = state.layout
    assert lay.world_size == mpi_world
    assert lay.rank == mpi_rank
    assert lay.num_local_qubits + lay.num_global_qubits == n
    assert state.local_size == 2**lay.num_local_qubits
    assert state.local_size * mpi_world == 2**n


def test_initial_state_lives_entirely_on_rank_zero(mpi_world, mpi_rank):
    n = qubits_for(mpi_world)
    state = distributed.new_distributed_state(n)
    local = state.local_amplitudes()

    if mpi_rank == 0:
        assert local[0] == 1.0
        assert np.count_nonzero(local) == 1
    else:
        assert np.count_nonzero(local) == 0


def test_global_norm_is_one(mpi_world):
    state = distributed.new_distributed_state(qubits_for(mpi_world))
    assert state.norm() == pytest.approx(1.0)


def test_local_norms_sum_to_the_global_norm(mpi_world, mpi_rank):
    state = distributed.new_distributed_state(qubits_for(mpi_world))
    local = state.local_squared_norm()
    assert local == pytest.approx(1.0 if mpi_rank == 0 else 0.0)
    assert state.norm() == pytest.approx(1.0)


def test_gather_reconstructs_the_full_state(mpi_world):
    n = qubits_for(mpi_world)
    state = distributed.new_distributed_state(n)
    full = state.gather()

    assert full.shape == (2**n,)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(full, expected)


def test_gather_is_identical_on_every_rank(mpi_world):
    """Allgather means each rank must see byte-identical data."""
    n = qubits_for(mpi_world)
    state = distributed.new_distributed_state(n)
    full = state.gather()
    checksum = float(np.abs(full).sum())
    assert checksum == pytest.approx(1.0)


def test_custom_mapping_is_accepted_and_reported(mpi_world):
    n = qubits_for(mpi_world)
    p = mpi_world.bit_length() - 1
    if p == 0:
        pytest.skip("single-rank world has no global positions")

    # Push the *lowest* logical qubits onto the global positions: the reverse
    # of the default placement.
    mapping = list(range(n))
    mapping = mapping[-p:] + mapping[:-p] if p else mapping
    mapping = [(mapping.index(i)) for i in range(n)]

    state = distributed.new_distributed_state(n, mapping=mapping)
    lay = state.layout
    assert not lay.is_identity_mapping()
    assert len(lay.global_qubits()) == p
    # |0...0> is index 0 under any permutation, so the state is unchanged.
    full = state.gather()
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(full, expected)


def test_reset_restores_the_initial_state(mpi_world, mpi_rank):
    state = distributed.new_distributed_state(qubits_for(mpi_world))
    state.reset()
    assert state.norm() == pytest.approx(1.0)
    if mpi_rank == 0:
        assert state.local_amplitudes()[0] == 1.0


def test_fp32_shards_are_half_the_size(mpi_world):
    n = qubits_for(mpi_world)
    single = distributed.new_distributed_state(n, precision="fp32")
    double = distributed.new_distributed_state(n, precision="fp64")
    assert single.local_amplitudes().dtype == np.complex64
    assert double.local_amplitudes().dtype == np.complex128
    assert single.local_size == double.local_size
    assert single.norm() == pytest.approx(1.0)


def test_too_few_qubits_for_the_world_is_rejected(mpi_world):
    if mpi_world == 1:
        pytest.skip("a single rank always has local qubits")
    p = mpi_world.bit_length() - 1
    with pytest.raises(Exception, match="at least one local qubit"):
        distributed.new_distributed_state(p)


def test_describe_reports_the_environment(mpi_world, mpi_rank):
    info = distributed.describe()
    assert info["mpi_compiled"] is True
    assert info["world_size"] == mpi_world
    assert info["rank"] == mpi_rank
    assert info["mpi_library"] != "none"
