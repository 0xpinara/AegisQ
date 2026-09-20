"""Windowed placement on the live distributed runtime.

The plan's predicted traffic has to equal what actually crosses the network,
and the rewritten circuit has to compute the same state once its final slot
permutation is undone. Both are checked here, because a planner that only
agrees with itself is worth nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.algorithms import build_circuit
from aegisq.compiler import CommunicationCostModel
from aegisq.compiler.cost_model import mapping_from_global_qubits
from aegisq.compiler.dynamic_mapper import (
    apply_plan,
    permute_amplitudes,
    plan_dynamic_placement,
)
from aegisq.runtime import Simulator, distributed
from aegisq.runtime.native import to_native_circuit

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def width(mpi_world) -> int:
    return (mpi_world.bit_length() - 1) + 7


def measure(circuit, mapping=None):
    state = distributed.new_distributed_state(circuit.num_qubits, mapping=mapping)
    state.apply_circuit(to_native_circuit(circuit))
    return state


@pytest.mark.parametrize("family", ["qft", "ising", "random", "grover"])
def test_predicted_traffic_matches_measurement(mpi_world, width, family):
    circuit = build_circuit(family, width, **({"iterations": 2} if family == "grover" else {}))
    model = CommunicationCostModel(circuit.num_qubits, mpi_world)
    plan = plan_dynamic_placement(circuit, model, window_size=48)

    static_mapping = mapping_from_global_qubits(circuit.num_qubits, plan.static_assignment)
    assert measure(circuit, static_mapping).reduced_metrics()["bytes_sent"] == plan.static_bytes

    rewritten = apply_plan(circuit, plan, model)
    # Slots are baked into the operands, so the runtime keeps the identity map.
    assert measure(rewritten).reduced_metrics()["bytes_sent"] == plan.dynamic_bytes


@pytest.mark.parametrize("family", ["qft", "random", "grover"])
def test_rewritten_circuit_is_exact_after_relabelling(mpi_world, width, family):
    circuit = build_circuit(family, width, **({"iterations": 2} if family == "grover" else {}))
    model = CommunicationCostModel(circuit.num_qubits, mpi_world)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    rewritten = apply_plan(circuit, plan, model)

    gathered = measure(rewritten).gather()
    logical = permute_amplitudes(gathered, plan.final_slots)
    expected = Simulator("reference").run(circuit).statevector
    assert np.allclose(logical, expected, atol=1e-10)


def test_restoration_in_circuit_needs_no_relabelling(mpi_world, width):
    circuit = build_circuit("qft", width)
    model = CommunicationCostModel(circuit.num_qubits, mpi_world)
    plan = plan_dynamic_placement(circuit, model, window_size=48, restore_order=True)
    rewritten = apply_plan(circuit, plan, model)

    expected = Simulator("reference").run(circuit).statevector
    assert np.allclose(measure(rewritten).gather(), expected, atol=1e-10)
    assert measure(rewritten).reduced_metrics()["bytes_sent"] == plan.dynamic_bytes


def test_windowing_reduces_measured_traffic_where_it_claims_to(mpi_world, width):
    if mpi_world == 1:
        pytest.skip("a single rank never communicates")
    circuit = build_circuit("grover", width, iterations=2)
    model = CommunicationCostModel(circuit.num_qubits, mpi_world)
    plan = plan_dynamic_placement(circuit, model, window_size=48)
    if not plan.worthwhile():
        pytest.skip("no predicted improvement for this configuration")

    static_mapping = mapping_from_global_qubits(circuit.num_qubits, plan.static_assignment)
    static_bytes = measure(circuit, static_mapping).reduced_metrics()["bytes_sent"]
    dynamic_bytes = measure(apply_plan(circuit, plan, model)).reduced_metrics()["bytes_sent"]

    assert dynamic_bytes < static_bytes
    measured = (static_bytes - dynamic_bytes) / static_bytes
    assert measured == pytest.approx(plan.improvement, abs=1e-12)
