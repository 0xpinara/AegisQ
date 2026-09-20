"""Gate fusion: a second, independent lever on communication."""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.algorithms import build_circuit, random_circuit
from aegisq.circuit import Circuit
from aegisq.circuit.gates import Gate, single_qubit_matrix
from aegisq.compiler import CommunicationCostModel, fuse, fuse_single_qubit_runs
from aegisq.compiler.cost_model import default_global_qubits
from aegisq.runtime import Simulator


def state(circuit: Circuit, backend: str = "cpp") -> np.ndarray:
    return Simulator(backend).run(circuit).statevector


# -- exactness --------------------------------------------------------------


@pytest.mark.parametrize("family", ["ghz", "qft", "ising", "random", "grover"])
def test_fusion_preserves_the_state_exactly(family):
    """Not up to a global phase: exactly, so comparisons stay elementwise."""
    circuit = build_circuit(family, 8)
    assert np.allclose(state(fuse(circuit)), state(circuit), atol=1e-13)


@pytest.mark.parametrize("seed", range(8))
def test_fusion_preserves_random_circuits(seed):
    circuit = random_circuit(6, depth=12, seed=seed)
    assert np.allclose(state(fuse(circuit)), state(circuit), atol=1e-13)


@pytest.mark.parametrize("backend", ["reference", "cpp"])
def test_both_backends_execute_fused_gates_identically(backend):
    circuit = random_circuit(5, depth=10, seed=3)
    fused = fuse(circuit)
    assert np.allclose(state(fused, backend), state(circuit, backend), atol=1e-13)


def test_a_run_of_inverses_fuses_to_the_identity():
    circuit = Circuit(2).h(0).h(0).cx(0, 1)
    result = fuse_single_qubit_runs(circuit)
    assert result.stats.runs_fused == 1
    fused_gate = result.circuit.gates[0]
    assert fused_gate.opcode == "u"
    assert np.allclose(single_qubit_matrix(fused_gate), np.eye(2), atol=1e-15)
    # An identity is diagonal, so it also stops costing communication.
    assert fused_gate.is_diagonal


def test_a_diagonal_run_stays_diagonal():
    circuit = Circuit(2).z(0).s(0).rz(0, 0.4).t(0).cx(0, 1)
    result = fuse_single_qubit_runs(circuit)
    assert result.stats.diagonal_runs == 1
    assert result.circuit.gates[0].is_diagonal


def test_a_mixed_run_is_not_diagonal():
    circuit = Circuit(2).rz(0, 0.3).h(0).cx(0, 1)
    assert not fuse_single_qubit_runs(circuit).circuit.gates[0].is_diagonal


# -- structure --------------------------------------------------------------


def test_runs_are_flushed_at_two_qubit_gates():
    circuit = Circuit(2).h(0).t(0).cx(0, 1).h(0).t(0)
    fused = fuse(circuit)
    assert [g.opcode for g in fused] == ["u", "cx", "u"]


def test_gates_on_other_qubits_do_not_break_a_run():
    """Single-qubit gates on different qubits commute, so the run continues."""
    circuit = Circuit(2).h(0).h(1).t(0)
    result = fuse_single_qubit_runs(circuit)
    assert result.stats.runs_fused == 1  # only qubit 0 has a run of two
    assert np.allclose(state(result.circuit), state(circuit), atol=1e-14)


def test_short_runs_are_left_alone():
    circuit = Circuit(2).h(0).cx(0, 1).t(1)
    fused = fuse(circuit)
    assert [g.opcode for g in fused] == ["h", "cx", "t"]


def test_min_run_controls_the_threshold():
    circuit = Circuit(1).h(0).t(0).s(0)
    assert [g.opcode for g in fuse(circuit, min_run=2)] == ["u"]
    assert [g.opcode for g in fuse(circuit, min_run=4)] == ["h", "t", "s"]
    with pytest.raises(ValueError, match="at least 2"):
        fuse(circuit, min_run=1)


def test_measurements_survive_fusion():
    circuit = Circuit(3).h(0).t(0).measure(0).measure(2)
    assert fuse(circuit).measured_qubits == (0, 2)


def test_empty_and_single_gate_circuits_are_unchanged():
    assert len(fuse(Circuit(3))) == 0
    assert [g.opcode for g in fuse(Circuit(2).h(0))] == ["h"]


def test_statistics_describe_the_pass():
    circuit = build_circuit("grover", 8)
    stats = fuse_single_qubit_runs(circuit).stats
    assert stats.gates_after < stats.gates_before
    assert stats.gates_absorbed >= 2 * stats.runs_fused
    assert stats.longest_run >= 2
    assert 0.0 < stats.reduction < 1.0
    assert "fused" in stats.summary()
    assert stats.as_dict()["gates_before"] == stats.gates_before


# -- interaction with the cost model ---------------------------------------


@pytest.mark.parametrize("family", ["qft", "ising", "random", "grover"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_fusion_never_increases_predicted_communication(family, world):
    """Fusing can only merge exchanges, never create one."""
    circuit = build_circuit(family, 12)
    model = CommunicationCostModel(circuit.num_qubits, world)
    placement = default_global_qubits(circuit.num_qubits, world)
    before = model.estimate(circuit, placement).bytes_sent
    after = model.estimate(fuse(circuit), placement).bytes_sent
    assert after <= before


def test_fusion_and_placement_compose():
    """The two levers address different redundancies, so they combine."""
    circuit = build_circuit("grover", 14)
    world = 8
    model = CommunicationCostModel(circuit.num_qubits, world)
    default = default_global_qubits(circuit.num_qubits, world)

    from aegisq.compiler import optimize_placement

    baseline = model.estimate(circuit, default).bytes_sent
    fused_only = model.estimate(fuse(circuit), default).bytes_sent
    placed_only = model.estimate(circuit, optimize_placement(circuit, world).global_qubits)
    fused = fuse(circuit)
    both = model.estimate(fused, optimize_placement(fused, world).global_qubits).bytes_sent

    assert fused_only < baseline
    assert placed_only.bytes_sent < baseline
    assert both < min(fused_only, placed_only.bytes_sent)


def test_a_fused_diagonal_run_on_a_global_qubit_is_free():
    n, world = 6, 4
    circuit = Circuit(n, name="diagonal-global")
    for qubit in range(n):
        circuit.h(qubit)
    for _ in range(10):
        circuit.rz(n - 1, 0.3)
        circuit.s(n - 1)

    model = CommunicationCostModel(n, world)
    placement = default_global_qubits(n, world)
    assert (
        model.estimate(circuit, placement).bytes_sent
        == model.estimate(fuse(circuit), placement).bytes_sent
    )


def test_fused_gates_are_rejected_by_the_qasm_emitter():
    from aegisq.circuit.qasm import to_qasm

    with pytest.raises(ValueError, match="no OpenQASM representation"):
        to_qasm(fuse(Circuit(1).h(0).t(0)))


def test_non_unitary_fused_gate_is_rejected():
    from aegisq.circuit import CircuitError

    with pytest.raises(CircuitError, match="not unitary"):
        Circuit(1).append(Gate("u", (0,), (1.0, 0, 1.0, 0, 0, 0, 1.0, 0)))
