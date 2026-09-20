"""Phase 10: the optimised placement must be real, not just predicted.

Two properties are checked on the live distributed runtime:

1. the optimised mapping produces the *same state* as the default placement
   (a faster wrong answer is not an optimisation), and
2. the measured byte counts match the cost model for both placements, so the
   reported reduction is a reduction in bytes that actually crossed the
   network.
"""

from __future__ import annotations

import pytest

from aegisq.algorithms import build_circuit
from aegisq.compiler import optimize_placement
from aegisq.compiler.cost_model import mapping_from_global_qubits
from aegisq.runtime import distributed
from aegisq.runtime.native import to_native_circuit

pytestmark = pytest.mark.mpi


def run_with_mapping(circuit, mapping=None):
    state = distributed.new_distributed_state(circuit.num_qubits, mapping=mapping)
    state.apply_circuit(to_native_circuit(circuit))
    return state


@pytest.fixture(scope="module")
def width(mpi_world) -> int:
    return (mpi_world.bit_length() - 1) + 6


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random"])
def test_optimized_mapping_computes_the_same_state(mpi_world, width, family):
    import numpy as np

    from aegisq.runtime import Simulator

    circuit = build_circuit(family, width)
    result = optimize_placement(circuit, world_size=mpi_world)

    expected = Simulator("reference").run(circuit).statevector
    baseline_state = run_with_mapping(circuit).gather()
    optimized_state = run_with_mapping(circuit, list(result.mapping)).gather()

    assert np.allclose(baseline_state, expected, atol=1e-10)
    assert np.allclose(optimized_state, expected, atol=1e-10)


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random", "grover"])
def test_measured_traffic_matches_the_prediction_for_both_placements(mpi_world, width, family):
    circuit = build_circuit(family, width)
    result = optimize_placement(circuit, world_size=mpi_world)

    baseline_metrics = run_with_mapping(circuit).reduced_metrics()
    optimized_metrics = run_with_mapping(circuit, list(result.mapping)).reduced_metrics()

    assert baseline_metrics["bytes_sent"] == result.baseline.bytes_sent
    assert optimized_metrics["bytes_sent"] == result.optimized.bytes_sent
    assert baseline_metrics["pairwise_exchanges"] == result.baseline.pairwise_exchanges
    assert optimized_metrics["pairwise_exchanges"] == result.optimized.pairwise_exchanges


def test_predicted_reduction_is_realised_on_the_wire(mpi_world, width):
    """For a circuit where the model predicts a saving, measure the saving."""
    if mpi_world == 1:
        pytest.skip("a single rank never communicates")

    circuit = build_circuit("random", width, seed=5)
    result = optimize_placement(circuit, world_size=mpi_world)
    if not result.improved():
        pytest.skip("no predicted improvement for this circuit and rank count")

    baseline = run_with_mapping(circuit).reduced_metrics()["bytes_sent"]
    optimized = run_with_mapping(circuit, list(result.mapping)).reduced_metrics()["bytes_sent"]

    assert optimized < baseline
    measured_reduction = (baseline - optimized) / baseline
    assert measured_reduction == pytest.approx(result.reduction, abs=1e-12)


def test_a_predicted_free_placement_sends_nothing(mpi_world, width):
    if mpi_world == 1:
        pytest.skip("a single rank never communicates")
    from aegisq.circuit import Circuit

    p = mpi_world.bit_length() - 1
    n = width
    circuit = Circuit(n, name="free-placement")
    # The last p qubits only ever take part in diagonal operations.
    cheap = list(range(n - p, n))
    for q in range(n - p):
        circuit.h(q).rx(q, 0.4)
        if q + 1 < n - p:
            circuit.cx(q, q + 1)
    for q in cheap:
        circuit.rz(q, 0.3)
        circuit.cz(0, q)

    result = optimize_placement(circuit, world_size=mpi_world)
    assert result.optimized.bytes_sent == 0

    metrics = run_with_mapping(circuit, list(result.mapping)).reduced_metrics()
    assert metrics["bytes_sent"] == 0
    assert metrics["pairwise_exchanges"] == 0


def test_mapping_helper_and_optimizer_agree(mpi_world, width):
    circuit = build_circuit("ising", width)
    result = optimize_placement(circuit, world_size=mpi_world)
    rebuilt = mapping_from_global_qubits(circuit.num_qubits, result.global_qubits)
    assert list(result.mapping) == rebuilt
