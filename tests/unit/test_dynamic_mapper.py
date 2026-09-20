"""Windowed placement: changing the qubit assignment mid-circuit."""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.algorithms import build_circuit, random_circuit
from aegisq.circuit import Circuit
from aegisq.compiler import CommunicationCostModel
from aegisq.compiler.dynamic_mapper import (
    apply_plan,
    dynamic_placement,
    permute_amplitudes,
    permute_counts,
    plan_dynamic_placement,
    swap_cost,
    transition_cost,
    transition_swaps,
)
from aegisq.compiler.static_mapper import StaticCommunicationMapper
from aegisq.runtime import Simulator


def state(circuit: Circuit) -> np.ndarray:
    return Simulator("cpp").run(circuit).statevector


# -- the rewrite is exact ---------------------------------------------------


def rewrite_and_relabel(circuit: Circuit, model: CommunicationCostModel, **kwargs):
    """Run a plan end to end, undoing the final slot permutation classically."""
    plan = plan_dynamic_placement(circuit, model, **kwargs)
    rewritten = apply_plan(circuit, plan, model)
    return permute_amplitudes(state(rewritten), plan.final_slots), plan


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random", "grover"])
def test_rewritten_circuit_computes_the_same_state(family):
    circuit = build_circuit(family, 10)
    model = CommunicationCostModel(circuit.num_qubits, 4)
    got, _ = rewrite_and_relabel(circuit, model, window_size=32)
    assert np.allclose(got, state(circuit), atol=1e-11)


@pytest.mark.parametrize("family", ["qft", "random", "grover"])
def test_restoring_order_in_the_circuit_needs_no_relabelling(family):
    """With restore_order the circuit ends in logical order by itself."""
    circuit = build_circuit(family, 10)
    model = CommunicationCostModel(circuit.num_qubits, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=32, restore_order=True)
    rewritten = apply_plan(circuit, plan, model)
    assert plan.final_slots == tuple(range(circuit.num_qubits))
    assert np.allclose(state(rewritten), state(circuit), atol=1e-11)


@pytest.mark.parametrize("seed", range(6))
def test_rewritten_random_circuits_are_exact(seed):
    circuit = random_circuit(9, depth=14, seed=seed)
    model = CommunicationCostModel(circuit.num_qubits, 8)
    got, _ = rewrite_and_relabel(circuit, model, window_size=24)
    assert np.allclose(got, state(circuit), atol=1e-11)


def test_counts_are_relabelled_back_to_logical_order():
    """Checked on a deterministic outcome, where the answer is unambiguous.

    Counts from a permuted circuit are *not* expected to match shot for shot:
    the sampler walks amplitudes in index order, and permuting the qubits
    permutes that order (see docs/reproducibility.md). Relabelling is a
    bijection on outcome labels, so it is tested where the distribution has a
    single outcome.
    """
    marked = 0b101101
    circuit = Circuit(6, name="basis")
    for bit in range(6):
        if (marked >> bit) & 1:
            circuit.x(bit)
    circuit.measure_all()

    model = CommunicationCostModel(6, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=2)
    rewritten = apply_plan(circuit, plan, model)

    raw = Simulator("cpp").run(rewritten, shots=64, seed=1).counts
    relabelled = permute_counts(raw, plan.final_slots, circuit.measured_qubits)
    assert relabelled == {format(marked, "06b"): 64}


def test_relabelled_counts_sample_the_same_distribution():
    """For a spread distribution, compare probabilities rather than shots."""
    circuit = build_circuit("random", 8, seed=3).measure_all()
    model = CommunicationCostModel(8, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=24)
    rewritten = apply_plan(circuit, plan, model)

    shots = 20000
    raw = Simulator("cpp").run(rewritten, shots=shots, seed=1).counts
    relabelled = permute_counts(raw, plan.final_slots, circuit.measured_qubits)

    exact = np.abs(state(circuit)) ** 2
    assert sum(relabelled.values()) == shots
    for bitstring, count in relabelled.items():
        index = int(bitstring, 2)
        expected = exact[index] * shots
        assert abs(count - expected) < 5 * max(1.0, np.sqrt(max(expected, 1.0))) + 5


def test_measurement_follows_the_qubit_into_its_final_slot():
    circuit = build_circuit("qft", 8).measure(0).measure(3)
    model = CommunicationCostModel(8, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=32)
    rewritten = apply_plan(circuit, plan, model)
    assert rewritten.measured_qubits == tuple(sorted(plan.final_slots[q] for q in (0, 3)))


# -- the plan is honest about its own cost ---------------------------------


def test_the_baseline_is_the_best_static_assignment_not_the_default():
    """Otherwise windowing would take credit for what static placement does."""
    circuit = build_circuit("qft", 12)
    model = CommunicationCostModel(12, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    best_static = StaticCommunicationMapper(model).optimize(circuit)
    assert plan.static_assignment == best_static.global_qubits
    assert plan.static_bytes == best_static.optimized.bytes_sent


def test_transition_cost_counts_only_group_changes():
    model = CommunicationCostModel(10, 4)
    half_shard = (model.shard_bytes // 2) * model.world_size
    assert transition_cost(model, (8, 9), (8, 9)) == 0
    assert transition_cost(model, (8, 9), (8, 7)) == half_shard
    assert transition_cost(model, (8, 9), (0, 1)) == 2 * half_shard


def test_swap_cost_is_free_only_between_two_local_slots():
    model = CommunicationCostModel(10, 4)  # eight local slots
    half_shard = (model.shard_bytes // 2) * model.world_size
    assert swap_cost(model, 0, 1) == 0
    assert swap_cost(model, 0, 8) == half_shard
    assert swap_cost(model, 8, 9) == half_shard


def test_transition_swaps_realise_the_permutation():
    current = [0, 1, 2, 3]
    target = [2, 3, 0, 1]
    slots = list(current)
    for a, b in transition_swaps(current, target):
        i, j = slots.index(a), slots.index(b)
        slots[i], slots[j] = slots[j], slots[i]
    assert slots == target


def test_reported_cost_uses_the_swaps_that_will_actually_be_emitted():
    """The planner and the rewriter must not disagree about the bill.

    An earlier version costed only group changes and so missed the
    global-to-global swaps a restoration needs, under-predicting its own
    traffic.
    """
    circuit = build_circuit("qft", 11)
    model = CommunicationCostModel(11, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    rewritten = apply_plan(circuit, plan, model)

    # Cost the rewritten circuit directly: slots are baked in, so the runtime
    # mapping is the identity.
    from aegisq.compiler.cost_model import default_global_qubits

    direct = model.estimate(rewritten, default_global_qubits(model.num_qubits, model.world_size))
    assert direct.bytes_sent == plan.dynamic_bytes


def test_reported_cost_is_exact_with_restoration_too():
    circuit = build_circuit("qft", 11)
    model = CommunicationCostModel(11, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=48, restore_order=True)
    rewritten = apply_plan(circuit, plan, model)

    from aegisq.compiler.cost_model import default_global_qubits

    direct = model.estimate(rewritten, default_global_qubits(model.num_qubits, model.world_size))
    assert direct.bytes_sent == plan.dynamic_bytes


def test_plan_totals_add_up():
    circuit = build_circuit("random", 10, seed=4)
    model = CommunicationCostModel(10, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=32)
    assert plan.dynamic_bytes == (
        sum(plan.window_bytes) + sum(plan.transition_bytes) + plan.restore_bytes
    )
    assert len(plan.assignments) == len(plan.windows)
    assert plan.switches <= len(plan.windows)


# -- when it helps, and when it does not -----------------------------------


@pytest.mark.parametrize("family", ["qft", "random", "grover"])
def test_windowing_beats_the_best_static_assignment_on_phased_circuits(family):
    circuit = build_circuit(family, 12, **({"iterations": 2} if family == "grover" else {}))
    model = CommunicationCostModel(12, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    assert plan.dynamic_bytes < plan.static_bytes
    assert plan.switches >= 1


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random", "grover"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_a_plan_is_never_worse_than_staying_put(family, world):
    """Holding the static assignment for every window is always available."""
    circuit = build_circuit(family, 12, **({"iterations": 2} if family == "grover" else {}))
    model = CommunicationCostModel(12, world)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    assert plan.dynamic_bytes <= plan.static_bytes


def test_a_uniform_circuit_gains_nothing_and_says_so():
    """GHZ has no phase structure, so the plan should not switch."""
    circuit = build_circuit("ghz", 12)
    model = CommunicationCostModel(12, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=32)
    assert plan.dynamic_bytes >= plan.static_bytes * 0.999
    assert not plan.worthwhile() or plan.improvement < 0.01


def test_dynamic_placement_never_returns_a_worse_circuit():
    """If windowing does not pay, the original circuit comes back unchanged."""
    circuit = build_circuit("ghz", 10)
    rewritten, plan = dynamic_placement(circuit, world_size=4, window_size=16)
    if not plan.worthwhile():
        assert rewritten is circuit


def test_single_rank_planning_is_trivial():
    circuit = build_circuit("qft", 8)
    model = CommunicationCostModel(8, 1)
    plan = plan_dynamic_placement(circuit, model)
    assert plan.dynamic_bytes == 0
    assert plan.static_bytes == 0
    assert plan.windows == []


def test_empty_circuit_is_handled():
    model = CommunicationCostModel(6, 4)
    plan = plan_dynamic_placement(Circuit(6), model)
    assert plan.dynamic_bytes == 0
    assert apply_plan(Circuit(6), plan, model).num_qubits == 6


def test_window_size_must_be_positive():
    model = CommunicationCostModel(6, 4)
    with pytest.raises(ValueError, match="at least one gate"):
        plan_dynamic_placement(build_circuit("ghz", 6), model, window_size=0)


def test_width_mismatch_is_rejected():
    model = CommunicationCostModel(6, 4)
    with pytest.raises(ValueError, match="does not match"):
        plan_dynamic_placement(build_circuit("ghz", 8), model)


def test_report_states_when_static_wins():
    circuit = build_circuit("ghz", 10)
    model = CommunicationCostModel(10, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=4)
    text = plan.report()
    assert "windowed assignment" in text
    if not plan.worthwhile():
        assert "static assignment wins" in text


def test_plan_is_json_serialisable():
    import json

    circuit = build_circuit("random", 10, seed=1)
    model = CommunicationCostModel(10, 4)
    plan = plan_dynamic_placement(circuit, model, window_size=32)
    payload = json.loads(json.dumps(plan.as_dict()))
    assert payload["windows"] == len(plan.windows)
    assert payload["dynamic_bytes"] == plan.dynamic_bytes
