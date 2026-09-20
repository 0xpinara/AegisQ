"""Phase 9: the analytical communication cost model.

These tests fix the model's rules in closed form. The companion MPI suite
(`tests/mpi/test_cost_model_prediction.py`) checks the same rules against
what the runtime actually sends.
"""

from __future__ import annotations

import pytest

from aegisq.circuit import Circuit, Gate
from aegisq.compiler.cost_model import (
    CommunicationCostModel,
    default_global_qubits,
    mapping_from_global_qubits,
)
from tests.conftest import random_circuit


@pytest.fixture
def model() -> CommunicationCostModel:
    return CommunicationCostModel(num_qubits=10, world_size=4, precision="fp64")


def test_shard_geometry(model):
    assert model.num_global_qubits == 2
    assert model.num_local_qubits == 8
    assert model.shard_amplitudes == 2**8
    assert model.shard_bytes == 2**8 * 16


@pytest.mark.parametrize("opcode, params", [("z", ()), ("s", ()), ("t", ()), ("rz", (0.4,))])
def test_diagonal_gates_are_free_anywhere(model, opcode, params):
    globals_ = frozenset({8, 9})
    assert model.gate_cost(Gate(opcode, (9,), params), globals_).bytes_sent == 0
    assert model.gate_cost(Gate(opcode, (0,), params), globals_).bytes_sent == 0


def test_cz_is_free_in_every_placement(model):
    globals_ = frozenset({8, 9})
    for pair in [(0, 1), (0, 9), (9, 0), (8, 9)]:
        assert model.gate_cost(Gate("cz", pair), globals_).bytes_sent == 0


@pytest.mark.parametrize("opcode, params", [("x", ()), ("y", ()), ("h", ()), ("rx", (0.2,))])
def test_nondiagonal_single_qubit_costs_a_full_shard_per_rank(model, opcode, params):
    globals_ = frozenset({8, 9})
    cost = model.gate_cost(Gate(opcode, (9,), params), globals_)
    assert cost.message_bytes == model.shard_bytes
    assert cost.exchanges == model.world_size
    assert cost.bytes_sent == model.shard_bytes * model.world_size
    # Local placement is free.
    assert model.gate_cost(Gate(opcode, (0,), params), globals_).bytes_sent == 0


def test_cx_with_local_target_is_free_whatever_the_control(model):
    globals_ = frozenset({8, 9})
    assert model.gate_cost(Gate("cx", (0, 1)), globals_).bytes_sent == 0
    assert model.gate_cost(Gate("cx", (9, 1)), globals_).bytes_sent == 0


def test_cx_with_local_control_and_global_target_costs_half_a_shard(model):
    cost = model.gate_cost(Gate("cx", (0, 9)), frozenset({8, 9}))
    assert cost.message_bytes == model.shard_bytes // 2
    assert cost.exchanges == model.world_size
    assert cost.bytes_sent == model.shard_bytes // 2 * model.world_size


def test_cx_between_global_qubits_moves_the_same_bytes_in_fewer_messages(model):
    globals_ = frozenset({8, 9})
    local_control = model.gate_cost(Gate("cx", (0, 9)), globals_)
    global_control = model.gate_cost(Gate("cx", (8, 9)), globals_)
    assert global_control.bytes_sent == local_control.bytes_sent
    assert global_control.exchanges == local_control.exchanges // 2
    assert global_control.message_bytes == 2 * local_control.message_bytes


def test_swap_cost_depends_on_how_many_operands_are_global(model):
    globals_ = frozenset({8, 9})
    assert model.gate_cost(Gate("swap", (0, 1)), globals_).bytes_sent == 0
    one = model.gate_cost(Gate("swap", (0, 9)), globals_)
    both = model.gate_cost(Gate("swap", (8, 9)), globals_)
    assert one.bytes_sent == model.shard_bytes // 2 * model.world_size
    assert both.bytes_sent == one.bytes_sent
    assert both.exchanges == one.exchanges // 2


def test_fp32_halves_the_predicted_traffic():
    circuit = Circuit(10).h(9)
    big = CommunicationCostModel(10, 4, "fp64").estimate(circuit, [8, 9])
    small = CommunicationCostModel(10, 4, "fp32").estimate(circuit, [8, 9])
    assert small.bytes_sent * 2 == big.bytes_sent


def test_single_rank_world_never_communicates():
    model = CommunicationCostModel(8, 1)
    circuit = random_circuit(8, depth=10, seed=1)
    estimate = model.estimate(circuit, [])
    assert estimate.bytes_sent == 0
    assert estimate.pairwise_exchanges == 0


def test_default_placement_selects_the_highest_qubits():
    assert default_global_qubits(10, 4) == (8, 9)
    assert default_global_qubits(10, 8) == (7, 8, 9)
    assert default_global_qubits(10, 1) == ()


def test_mapping_helper_places_the_chosen_qubits_globally():
    mapping = mapping_from_global_qubits(6, [1, 4])
    # 4 local positions (0..3) then the two global ones.
    assert mapping[1] == 4
    assert mapping[4] == 5
    assert sorted(mapping) == list(range(6))


def test_estimate_for_mapping_matches_estimate_for_the_same_set(model):
    circuit = random_circuit(10, depth=8, seed=3)
    mapping = mapping_from_global_qubits(10, [2, 7])
    assert (
        model.estimate_for_mapping(circuit, mapping).bytes_sent
        == model.estimate(circuit, [2, 7]).bytes_sent
    )


@pytest.mark.parametrize("seed", range(12))
def test_aggregated_fast_path_agrees_with_per_gate_accounting(seed):
    """The optimiser's fast scorer must not drift from the readable rules."""
    model = CommunicationCostModel(12, 8, "fp64")
    circuit = random_circuit(12, depth=12, seed=seed)
    profile = model.compile_circuit(circuit)

    for globals_ in [(9, 10, 11), (0, 1, 2), (0, 5, 11), (3, 4, 7)]:
        estimate = model.estimate(circuit, globals_)
        assert profile.bytes_for(globals_) == estimate.bytes_sent
        assert profile.exchanges_for(globals_) == estimate.pairwise_exchanges


def test_per_opcode_breakdown_sums_to_the_total():
    model = CommunicationCostModel(12, 4)
    circuit = random_circuit(12, depth=10, seed=7)
    estimate = model.estimate(circuit, [10, 11])
    assert sum(v["bytes_sent"] for v in estimate.per_opcode.values()) == estimate.bytes_sent
    assert sum(v["gates"] for v in estimate.per_opcode.values()) == len(circuit)


def test_wrong_number_of_global_qubits_is_rejected(model):
    with pytest.raises(ValueError, match="exactly 2"):
        model.estimate(Circuit(10).h(0), [1])


def test_non_power_of_two_world_is_rejected():
    with pytest.raises(ValueError, match="power of two"):
        CommunicationCostModel(10, 6)


def test_too_many_ranks_for_the_circuit_is_rejected():
    with pytest.raises(ValueError, match="no local qubits"):
        CommunicationCostModel(2, 4)


def test_placement_choice_can_change_predicted_traffic_by_a_large_factor():
    """A QFT-like chain concentrates communication on its target qubits."""
    n = 14
    circuit = Circuit(n, name="chain")
    for q in range(n):
        circuit.h(q)
    for q in range(n - 1):
        circuit.cx(q, q + 1)

    model = CommunicationCostModel(n, 4)
    worst = min(
        (model.estimate(circuit, g) for g in [(12, 13), (0, 1), (5, 6)]),
        key=lambda e: -e.bytes_sent,
    )
    best = min(
        (model.estimate(circuit, g) for g in [(12, 13), (0, 1), (5, 6)]),
        key=lambda e: e.bytes_sent,
    )
    assert best.bytes_sent < worst.bytes_sent
