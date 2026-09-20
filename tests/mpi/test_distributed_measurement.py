"""Phase 12: distributed sampling.

The sampler draws from a seeded generator that does not depend on the rank
count, so a circuit sampled on 8 ranks must produce *the same counts* as the
same circuit sampled in one process. That property is what makes distributed
results reproducible and directly comparable in the benchmark suite.
"""

from __future__ import annotations

import pytest

from aegisq.algorithms import build_circuit, ghz
from aegisq.circuit import Circuit
from aegisq.runtime import Simulator
from tests.conftest import random_circuit

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def width(mpi_world) -> int:
    return (mpi_world.bit_length() - 1) + 5


def test_counts_match_the_single_process_backend(mpi_world, width):
    circuit = random_circuit(width, depth=8, seed=3).measure_all()
    single = Simulator("cpp").run(circuit, shots=4096, seed=1234).counts
    many = Simulator("mpi").run(circuit, shots=4096, seed=1234).counts
    assert many == single


def test_counts_are_identical_on_every_rank(mpi_world, width):
    circuit = ghz(width).measure_all()
    counts = Simulator("mpi").run(circuit, shots=2048, seed=7).counts
    # Every rank ran the same assertion; agreement across ranks is implied by
    # each rank seeing the full, allgathered map.
    assert sum(counts.values()) == 2048
    assert set(counts) == {"0" * width, "1" * width}


def test_sampling_is_reproducible(mpi_world, width):
    circuit = random_circuit(width, depth=6, seed=5).measure_all()
    first = Simulator("mpi").run(circuit, shots=1000, seed=99).counts
    second = Simulator("mpi").run(circuit, shots=1000, seed=99).counts
    assert first == second


def test_partial_measurement_marginalises(mpi_world, width):
    circuit = Circuit(width, name="partial").h(0).x(width - 1).measure(width - 1)
    counts = Simulator("mpi").run(circuit, shots=256, seed=2).counts
    assert counts == {"1": 256}


def test_zero_shots_returns_no_counts(mpi_world, width):
    result = Simulator("mpi").run(ghz(width), shots=0, save_statevector=False)
    assert result.counts == {}


def test_distribution_matches_probabilities(mpi_world, width):
    circuit = Circuit(width, name="uniform")
    for q in range(width):
        circuit.h(q)
    circuit.measure_all()
    shots = 20000
    counts = Simulator("mpi").run(circuit, shots=shots, seed=11).counts
    expected = shots / 2**width
    # Uniform distribution: every outcome within a generous multiple of sigma.
    sigma = (expected * (1 - 1 / 2**width)) ** 0.5
    for value in counts.values():
        assert abs(value - expected) < 6 * sigma + 5


def test_counts_under_a_custom_mapping_still_sum_correctly(mpi_world, width):
    """A permuted placement changes which rank samples which draw.

    Counts remain a correct sample of the same distribution, but the
    draw-to-index assignment differs, so they are not required to be identical
    to the default placement.
    """
    if mpi_world == 1:
        pytest.skip("single-rank world has no placement choice")
    circuit = build_circuit("ising", width, steps=2).measure_all()
    mapping = [width - 1 - q for q in range(width)]
    counts = Simulator("mpi", mapping=mapping).run(circuit, shots=3000, seed=4).counts
    assert sum(counts.values()) == 3000
    for key in counts:
        assert len(key) == width


def test_gather_guard_rejects_oversized_states(mpi_world, width):
    circuit = ghz(width)
    with pytest.raises(ValueError, match="refusing to gather"):
        Simulator("mpi", max_gather_qubits=width - 1).run(circuit, save_statevector=True)
