"""Phase 1: circuit IR construction, validation and serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.circuit import Circuit, CircuitError, Gate
from aegisq.circuit.gates import inverse, single_qubit_matrix, two_qubit_matrix


def test_builder_records_gates_in_order():
    c = Circuit(3, name="demo")
    c.h(0).cx(0, 1).rz(2, 0.5)
    assert [g.opcode for g in c] == ["h", "cx", "rz"]
    assert c.gates[2].params == (0.5,)
    assert len(c) == 3


def test_depth_counts_longest_dependency_chain():
    c = Circuit(3)
    c.h(0).h(1).h(2)  # parallel layer
    assert c.depth() == 1
    c.cx(0, 1).cx(1, 2)  # sequential chain through qubit 1
    assert c.depth() == 3


def test_gate_role_classification():
    cx = Gate("cx", (2, 5))
    assert cx.control_qubits == (2,)
    assert cx.target_qubits == (5,)
    assert not cx.is_diagonal
    assert Gate("cz", (0, 1)).is_diagonal
    assert Gate("rz", (0,), (0.3,)).is_diagonal
    assert not Gate("rx", (0,), (0.3,)).is_diagonal


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda c: c.x(7), "out of range"),
        (lambda c: c.cx(1, 1), "distinct"),
        (lambda c: c.append(Gate("h", (0, 1))), "acts on 1 qubit"),
        (lambda c: c.append(Gate("rz", (0,))), "takes 1 parameter"),
        (lambda c: c.append(Gate("toffoli", (0, 1))), "unsupported operation"),
    ],
)
def test_invalid_instructions_are_rejected(build, message):
    c = Circuit(3)
    with pytest.raises(CircuitError, match=message):
        build(c)


def test_zero_qubit_circuit_rejected():
    with pytest.raises(CircuitError, match="at least one qubit"):
        Circuit(0)


def test_measurement_is_deduplicated_and_sorted():
    c = Circuit(4)
    c.measure(3).measure(1).measure(3)
    assert c.measured_qubits == (1, 3)


def test_dict_roundtrip_is_stable():
    c = Circuit(3, name="rt")
    c.h(0).cx(0, 1).ry(2, -0.25).measure_all()
    restored = Circuit.from_dict(c.to_dict())
    assert restored.to_dict() == c.to_dict()
    assert [str(g) for g in restored] == [str(g) for g in c]


def test_unknown_schema_rejected():
    with pytest.raises(CircuitError, match="schema"):
        Circuit.from_dict({"schema": "other.v9", "num_qubits": 1, "gates": []})


def test_inverse_reverses_and_daggers():
    c = Circuit(2)
    c.h(0).rz(1, 0.4).cx(0, 1)
    inv = c.inverse()
    assert [g.opcode for g in inv] == ["cx", "rz", "h"]
    assert inv.gates[1].params == (-0.4,)


def test_gate_inverse_matrices_are_adjoints_up_to_global_phase():
    # S and T are inverted through RZ, which is exact only up to a global
    # phase; that substitution is documented in aegisq.circuit.gates.inverse.
    for gate in (
        Gate("s", (0,)),
        Gate("t", (0,)),
        Gate("rx", (0,), (0.9,)),
        Gate("h", (0,)),
    ):
        product = single_qubit_matrix(gate) @ single_qubit_matrix(inverse(gate))
        phase = product[0, 0]
        assert abs(abs(phase) - 1.0) < 1e-12
        assert np.allclose(product / phase, np.eye(2), atol=1e-12)


def test_self_inverse_gates_are_exactly_involutions():
    for gate in (Gate("x", (0,)), Gate("y", (0,)), Gate("z", (0,)), Gate("h", (0,))):
        m = single_qubit_matrix(gate)
        assert np.allclose(m @ m, np.eye(2), atol=1e-12)


def test_two_qubit_matrices_are_unitary():
    for opcode in ("cx", "cz", "swap"):
        m = two_qubit_matrix(Gate(opcode, (0, 1)))
        assert np.allclose(m @ m.conj().T, np.eye(4), atol=1e-12)


def test_compose_requires_matching_width():
    with pytest.raises(CircuitError, match="compose"):
        Circuit(2).compose(Circuit(3))


def test_bool_is_not_accepted_as_a_qubit_index():
    """`True` is an int in Python, so the check has to exclude bool on purpose."""
    import pytest

    from aegisq.circuit.circuit import Circuit, CircuitError

    circuit = Circuit(2)
    with pytest.raises(CircuitError, match="must be an int"):
        circuit.h(True)


def test_bool_is_not_accepted_as_a_gate_parameter():
    import pytest

    from aegisq.circuit.circuit import Circuit, CircuitError

    circuit = Circuit(2)
    with pytest.raises(CircuitError, match="real number"):
        circuit.rz(0, True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_angles_are_rejected(value):
    """A NaN angle would propagate silently through the whole state vector."""
    from aegisq.circuit.circuit import Circuit, CircuitError

    circuit = Circuit(1)
    with pytest.raises(CircuitError, match="finite"):
        circuit.rz(0, value)


def test_compose_keeps_gates_and_measurements_from_both_sides():
    from aegisq.circuit.circuit import Circuit

    left = Circuit(3, name="left").h(0)
    left.measure(0)
    right = Circuit(3, name="right").cx(0, 1)
    right.measure(2)

    joined = left.compose(right, name="joined")
    assert [g.opcode for g in joined] == ["h", "cx"]
    assert sorted(joined.measured_qubits) == [0, 2]
    assert joined.name == "joined"


def test_compose_rejects_a_wider_circuit():
    import pytest

    from aegisq.circuit.circuit import Circuit, CircuitError

    with pytest.raises(CircuitError, match="cannot compose"):
        Circuit(2).compose(Circuit(3))


def test_printing_a_long_circuit_truncates():
    from aegisq.circuit.circuit import Circuit

    circuit = Circuit(1)
    for _ in range(25):
        circuit.h(0)
    text = str(circuit)
    assert "... 5 more" in text
    assert text.count("\n") == 21  # header plus twenty gates
