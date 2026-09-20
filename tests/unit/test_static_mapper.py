"""Phase 10: communication-aware qubit placement."""

from __future__ import annotations

import pytest

from aegisq.algorithms import build_circuit, ghz, ising_trotter, qft, random_circuit
from aegisq.circuit import Circuit
from aegisq.compiler import CommunicationCostModel, StaticCommunicationMapper, optimize_placement
from aegisq.compiler.cost_model import default_global_qubits


def test_single_rank_search_is_trivial():
    result = optimize_placement(ghz(8), world_size=1)
    assert result.global_qubits == ()
    assert result.baseline.bytes_sent == 0
    assert result.optimized.bytes_sent == 0
    assert result.strategy.startswith("trivial")
    assert result.optimal_for_cost_model


def test_exhaustive_search_is_used_for_small_spaces():
    result = optimize_placement(random_circuit(12, depth=10, seed=1), world_size=4)
    assert result.strategy == "exhaustive"
    assert result.candidates_evaluated == 66  # C(12, 2)
    assert result.optimal_for_cost_model


def test_heuristic_is_used_and_labelled_when_the_space_is_large():
    circuit = random_circuit(20, depth=6, seed=2)
    model = CommunicationCostModel(20, 8)
    mapper = StaticCommunicationMapper(model, candidate_budget=10)
    result = mapper.optimize(circuit)
    assert result.strategy.startswith("greedy")
    assert not result.optimal_for_cost_model
    assert "not guaranteed optimal" in result.report()


def test_optimizer_never_returns_a_worse_placement_than_the_baseline():
    for seed in range(8):
        circuit = random_circuit(12, depth=10, seed=seed)
        result = optimize_placement(circuit, world_size=4)
        assert result.optimized.bytes_sent <= result.baseline.bytes_sent
        assert result.reduction >= 0.0


def test_exhaustive_result_is_optimal_for_the_cost_model():
    """Brute force over the same space must not find anything better."""
    import itertools

    circuit = random_circuit(11, depth=10, seed=4)
    model = CommunicationCostModel(11, 4)
    result = StaticCommunicationMapper(model).optimize(circuit)

    best = min(
        model.estimate(circuit, candidate).bytes_sent
        for candidate in itertools.combinations(range(11), 2)
    )
    assert result.optimized.bytes_sent == best


def test_heuristic_gets_close_to_the_exhaustive_optimum():
    circuit = random_circuit(14, depth=10, seed=6)
    model = CommunicationCostModel(14, 8)
    exhaustive = StaticCommunicationMapper(model).optimize(circuit)
    heuristic = StaticCommunicationMapper(model, candidate_budget=1).optimize(circuit)
    assert heuristic.optimized.bytes_sent >= exhaustive.optimized.bytes_sent
    # The local search should stay within a sensible factor of the optimum.
    assert heuristic.optimized.bytes_sent <= exhaustive.optimized.bytes_sent * 1.5


def test_mapping_places_the_chosen_qubits_on_global_positions():
    circuit = random_circuit(10, depth=8, seed=5)
    result = optimize_placement(circuit, world_size=4)
    num_local = 10 - 2
    for qubit in result.global_qubits:
        assert result.mapping[qubit] >= num_local
    for qubit in range(10):
        if qubit not in result.global_qubits:
            assert result.mapping[qubit] < num_local
    assert sorted(result.mapping) == list(range(10))


def test_reported_mapping_reproduces_the_reported_estimate():
    circuit = random_circuit(12, depth=10, seed=9)
    model = CommunicationCostModel(12, 8)
    result = StaticCommunicationMapper(model).optimize(circuit)
    again = model.estimate_for_mapping(circuit, result.mapping)
    assert again.bytes_sent == result.optimized.bytes_sent
    assert again.pairwise_exchanges == result.optimized.pairwise_exchanges


def test_a_circuit_with_only_diagonal_global_gates_is_already_free():
    """If nothing can be saved the optimizer reports no reduction, not a fake one."""
    n = 8
    circuit = Circuit(n, name="diagonal-only")
    for q in range(n):
        circuit.rz(q, 0.3)
        circuit.cz(q, (q + 1) % n)
    result = optimize_placement(circuit, world_size=4)
    assert result.baseline.bytes_sent == 0
    assert result.optimized.bytes_sent == 0
    assert result.reduction == 0.0
    assert not result.improved()


def test_optimizer_finds_the_free_placement_when_one_exists():
    """One qubit is used only by diagonal gates; it belongs on a rank."""
    n = 6
    circuit = Circuit(n, name="one-cheap-qubit")
    for q in range(n - 1):
        circuit.h(q)
        circuit.cx(q, (q + 1) % (n - 1))
    # qubit n-1 only ever sees diagonal operations
    for _ in range(20):
        circuit.rz(n - 1, 0.2)
        circuit.cz(n - 1, 0)

    result = optimize_placement(circuit, world_size=2)
    assert result.global_qubits == (n - 1,)
    assert result.optimized.bytes_sent == 0
    assert result.baseline.bytes_sent >= 0


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random", "grover"])
def test_every_benchmark_family_can_be_optimized(family):
    circuit = build_circuit(family, 14)
    result = optimize_placement(circuit, world_size=4)
    assert result.optimized.bytes_sent <= result.baseline.bytes_sent
    assert len(result.global_qubits) == 2
    assert "Predicted reduction" in result.report()


def test_qft_default_placement_is_already_good():
    """An honest negative result: the QFT's structure favours high qubits.

    In this decomposition the controlled phases use the *lower*-indexed qubit
    as the CX target, so low qubits are expensive to place globally and the
    default placement is already near-optimal.
    """
    circuit = qft(14)
    result = optimize_placement(circuit, world_size=4)
    assert result.reduction < 0.2


def test_ising_placement_reduces_predicted_traffic():
    result = optimize_placement(ising_trotter(16, steps=4), world_size=8)
    assert result.optimized.bytes_sent < result.baseline.bytes_sent


def test_report_is_explicit_about_being_a_prediction():
    result = optimize_placement(random_circuit(10, depth=6, seed=1), world_size=4)
    assert "cost-model predictions" in result.report()


def test_as_dict_is_json_serialisable():
    import json

    result = optimize_placement(random_circuit(10, depth=6, seed=1), world_size=4)
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["global_qubits"] == list(result.global_qubits)
    assert payload["optimal_for_cost_model"] is True


def test_width_mismatch_is_rejected():
    model = CommunicationCostModel(10, 4)
    with pytest.raises(ValueError, match="cost model expects"):
        StaticCommunicationMapper(model).optimize(ghz(8))


def test_baseline_defaults_to_the_runtime_placement():
    circuit = random_circuit(10, depth=8, seed=2)
    result = optimize_placement(circuit, world_size=4)
    assert result.baseline.global_qubits == default_global_qubits(10, 4)
