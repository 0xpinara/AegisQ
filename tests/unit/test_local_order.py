"""Ordering the local qubits, which costs nothing in bytes."""

from __future__ import annotations

import random
from itertools import permutations

import pytest

from aegisq.circuit import Circuit
from aegisq.compiler.local_order import (
    hungarian,
    load_position_costs,
    optimise_local_order,
)

# Two positions cheap, one expensive, so the right answer is obvious by hand.
TOY_COSTS = {"cz": (10.0, 1.0, 1.0), "h": (1.0, 1.0, 1.0)}


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_assignment_matches_brute_force(size):
    """The solver is exact, so an exhaustive check is available at this size."""
    rng = random.Random(size)
    for _ in range(60):
        cost = [[rng.uniform(0, 10) for _ in range(size)] for _ in range(size)]
        chosen = hungarian(cost)
        assert sorted(chosen) == list(range(size)), "not a permutation"
        got = sum(cost[row][chosen[row]] for row in range(size))
        best = min(
            sum(cost[row][order[row]] for row in range(size)) for order in permutations(range(size))
        )
        assert got == pytest.approx(best)


def test_an_empty_problem_is_not_an_error():
    assert hungarian([]) == []


def test_the_costed_operand_of_a_two_qubit_gate_is_its_target():
    """The kernel sweep varies the target, so that is what gets priced.

    Pricing the control instead would move the wrong qubit, and with a
    symmetric gate like `cz` the mistake is invisible in the state.
    """
    # Target is qubit 0, which the identity puts on the expensive position.
    circuit = Circuit(4, name="targets")
    for _ in range(20):
        circuit.cz(3, 0)

    result = optimise_local_order(circuit, global_qubits=(), num_qubits=4, position_costs=TOY_COSTS)
    assert result.mapping[0] != 0, "the costed operand stayed on the slow position"
    assert result.predicted_saving > 0
    # It should end on the cheapest position available. The other three
    # qubits are never costed, so which of them takes the slow position is
    # arbitrary and not asserted.
    assert result.mapping[0] == 3


def test_global_qubits_keep_their_positions():
    """Moving a global qubit would change the traffic, which is not free."""
    circuit = Circuit(6, name="mixed")
    for q in range(5):
        circuit.cz(q, q + 1)

    result = optimise_local_order(
        circuit, global_qubits=(4, 5), num_qubits=6, position_costs=TOY_COSTS
    )
    assert result.mapping[4] == 4
    assert result.mapping[5] == 5
    assert sorted(result.mapping) == list(range(6)), "not a permutation"


def test_a_circuit_with_nothing_to_move_predicts_no_saving():
    """One local qubit leaves no ordering to choose."""
    circuit = Circuit(3, name="tiny").h(0)
    result = optimise_local_order(
        circuit, global_qubits=(1, 2), num_qubits=3, position_costs=TOY_COSTS
    )
    assert result.predicted_saving == 0.0
    assert result.mapping == (0, 1, 2)


def test_the_prediction_is_never_worse_than_leaving_it_alone():
    """The identity ordering is in the search space, so a loss is a bug."""
    rng = random.Random(7)
    costs = load_position_costs()
    for trial in range(10):
        circuit = Circuit(12, name=f"random{trial}")
        for _ in range(80):
            a, b = rng.sample(range(12), 2)
            circuit.cz(a, b) if rng.random() < 0.5 else circuit.h(a)
        result = optimise_local_order(
            circuit, global_qubits=(10, 11), num_qubits=12, position_costs=costs
        )
        assert result.predicted_seconds <= result.baseline_seconds + 1e-12
        assert result.predicted_saving >= 0.0


def test_reordering_locally_cannot_change_the_traffic():
    """The whole argument for doing this is that bytes do not notice.

    If a local permutation changed the predicted byte count, the feature
    would be trading network cost for local cost without saying so.
    """
    from aegisq.algorithms import build_circuit
    from aegisq.compiler import CommunicationCostModel, StaticCommunicationMapper

    circuit = build_circuit("random", 14, seed=3)
    model = CommunicationCostModel(14, 4)
    global_qubits = StaticCommunicationMapper(model).optimize(circuit).global_qubits

    before = model.estimate(circuit, global_qubits)
    result = optimise_local_order(circuit, global_qubits, 14)
    after = model.estimate(circuit, global_qubits)

    assert before.bytes_sent == after.bytes_sent
    assert before.pairwise_exchanges == after.pairwise_exchanges
    # And the global set really is untouched by the permutation.
    assert all(result.mapping[q] == q for q in global_qubits)
