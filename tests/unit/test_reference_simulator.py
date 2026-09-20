"""Phase 1: correctness of the NumPy reference state-vector simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import Simulator
from aegisq.runtime.reference import ReferenceStateVector

INV_SQRT2 = 1.0 / math.sqrt(2.0)


def run(circuit: Circuit, precision: str = "fp64") -> np.ndarray:
    state = ReferenceStateVector(circuit.num_qubits, precision=precision)
    state.apply_circuit(circuit)
    return state.state.copy()


def test_initial_state_is_all_zeros_basis_state():
    state = ReferenceStateVector(3)
    assert state.state[0] == 1.0
    assert np.count_nonzero(state.state) == 1
    assert state.norm() == pytest.approx(1.0)


def test_x_flips_qubit_zero(tol):
    c = Circuit(1).x(0)
    assert np.allclose(run(c), [0, 1], atol=tol)


def test_x_acts_on_the_requested_qubit_only(tol):
    # |000> --X(q1)--> |010>, i.e. basis index 2.
    c = Circuit(3).x(1)
    expected = np.zeros(8, dtype=complex)
    expected[2] = 1.0
    assert np.allclose(run(c), expected, atol=tol)


def test_hadamard_creates_uniform_superposition(tol):
    c = Circuit(1).h(0)
    assert np.allclose(run(c), [INV_SQRT2, INV_SQRT2], atol=tol)


def test_bell_state(tol):
    c = Circuit(2).h(0).cx(0, 1)
    expected = np.array([INV_SQRT2, 0, 0, INV_SQRT2])
    assert np.allclose(run(c), expected, atol=tol)


def test_ghz_state(tol):
    n = 5
    c = Circuit(n).h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    state = run(c)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = expected[-1] = INV_SQRT2
    assert np.allclose(state, expected, atol=tol)


def test_cnot_control_target_direction(tol):
    # control=q1, target=q0 with q1 set must flip q0 -> |11> = index 3.
    c = Circuit(2).x(1).cx(1, 0)
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    assert np.allclose(run(c), expected, atol=tol)


def test_cnot_does_nothing_when_control_is_zero(tol):
    c = Circuit(2).cx(0, 1)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(run(c), expected, atol=tol)


def test_swap_exchanges_amplitudes(tol):
    c = Circuit(2).x(0).swap(0, 1)
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0
    assert np.allclose(run(c), expected, atol=tol)


def test_cz_is_diagonal_and_phases_the_11_component(tol):
    c = Circuit(2).h(0).h(1).cz(0, 1)
    expected = 0.5 * np.array([1, 1, 1, -1], dtype=complex)
    assert np.allclose(run(c), expected, atol=tol)


@pytest.mark.parametrize("opcode", ["x", "y", "z", "h"])
def test_self_inverse_gates_return_to_start(opcode, tol):
    c = Circuit(3).h(0).h(1).cx(0, 2)  # non-trivial starting state
    reference = run(c)
    getattr(c, opcode)(1)
    getattr(c, opcode)(1)
    assert np.allclose(run(c), reference, atol=tol)


@pytest.mark.parametrize("opcode", ["rx", "ry", "rz"])
@pytest.mark.parametrize("theta", [0.0, 0.3, -1.7, math.pi])
def test_rotation_followed_by_its_inverse_is_identity(opcode, theta, tol):
    c = Circuit(2).h(0).cx(0, 1)
    reference = run(c)
    getattr(c, opcode)(1, theta)
    getattr(c, opcode)(1, -theta)
    assert np.allclose(run(c), reference, atol=tol)


def test_rz_only_changes_phases(tol):
    c = Circuit(1).h(0).rz(0, 0.7)
    state = run(c)
    assert np.allclose(np.abs(state), [INV_SQRT2, INV_SQRT2], atol=tol)


def test_norm_is_preserved_through_a_deep_random_circuit(tol):
    from tests.conftest import random_circuit

    c = random_circuit(6, depth=25, seed=11)
    state = ReferenceStateVector(c.num_qubits)
    state.apply_circuit(c)
    assert state.norm() == pytest.approx(1.0, abs=1e-12)


def test_norm_holds_after_every_gate():
    from tests.conftest import random_circuit

    c = random_circuit(4, depth=8, seed=3)
    state = ReferenceStateVector(c.num_qubits)
    for gate in c:
        state.apply(gate)
        assert state.norm() == pytest.approx(1.0, abs=1e-12)


def test_sampling_is_deterministic_for_a_fixed_seed():
    c = Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all()
    first = Simulator().run(c, shots=500, seed=1234).counts
    second = Simulator().run(c, shots=500, seed=1234).counts
    assert first == second
    assert set(first) == {"0000", "1111"}
    assert sum(first.values()) == 500


def test_different_seeds_generally_differ():
    c = Circuit(6)
    for q in range(6):
        c.h(q)
    c.measure_all()
    a = Simulator().run(c, shots=200, seed=1).counts
    b = Simulator().run(c, shots=200, seed=2).counts
    assert a != b


def test_partial_measurement_marginalises(tol):
    # q0 in superposition, q1 deterministic |1>: measuring q1 only yields "1".
    c = Circuit(2).h(0).x(1).measure(1)
    counts = Simulator().run(c, shots=64, seed=5).counts
    assert counts == {"1": 64}


def test_bitstring_order_puts_highest_qubit_first():
    c = Circuit(3).x(2).measure_all()
    counts = Simulator().run(c, shots=16, seed=0).counts
    assert counts == {"100": 16}


def test_zero_shots_returns_empty_counts():
    result = Simulator().run(Circuit(2).h(0), shots=0)
    assert result.counts == {}
    assert result.statevector is not None


def test_fp32_precision_tracks_fp64_loosely():
    c = Circuit(5).h(0).cx(0, 1).ry(2, 0.3).cz(1, 3).rx(4, -0.8)
    assert np.allclose(run(c, "fp32"), run(c, "fp64"), atol=1e-6)


def test_reference_refuses_absurd_allocations():
    with pytest.raises(MemoryError, match="refusing to allocate"):
        ReferenceStateVector(40)


def test_probability_distribution_sums_to_one():
    from tests.conftest import random_circuit

    c = random_circuit(5, depth=10, seed=99)
    state = ReferenceStateVector(c.num_qubits)
    state.apply_circuit(c)
    assert state.probabilities().sum() == pytest.approx(1.0)
    assert state.probabilities([0, 2]).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["reference", "cpp"])
def test_shots_are_validated_identically_on_every_backend(backend):
    """Validation lives in the front end, so the message cannot differ by backend."""
    circuit = Circuit(2).h(0)
    with pytest.raises(ValueError, match="shots must be non-negative"):
        Simulator(backend).run(circuit, shots=-1)
    with pytest.raises(TypeError, match="shots must be an int"):
        Simulator(backend).run(circuit, shots=2.5)
    with pytest.raises(TypeError, match="shots must be an int"):
        Simulator(backend).run(circuit, shots=True)


@pytest.mark.parametrize("backend", ["reference", "cpp"])
def test_seeds_are_validated(backend):
    circuit = Circuit(2).h(0)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        Simulator(backend).run(circuit, shots=4, seed=-1)
    with pytest.raises(TypeError, match="seed must be an int"):
        Simulator(backend).run(circuit, shots=4, seed="hello")
    # None is allowed and means "unseeded".
    assert Simulator(backend).run(circuit, shots=4, seed=None).counts


def test_unknown_backend_reports_available_ones():
    with pytest.raises(ValueError, match="available: .*reference"):
        Simulator(backend="quantum-teapot")
