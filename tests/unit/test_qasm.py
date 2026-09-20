"""Phase 12: the OpenQASM subset front end.

The parser accepts a documented subset and rejects everything else with a line
number. That is deliberate: a silently ignored instruction would produce a
wrong state and a benchmark that looks fine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aegisq.circuit.qasm import QasmError, parse_qasm, parse_qasm_file, to_qasm
from aegisq.runtime import Simulator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_minimal_circuit():
    circuit = parse_qasm("qreg q[2]; h q[0]; cx q[0], q[1];")
    assert circuit.num_qubits == 2
    assert [g.opcode for g in circuit] == ["h", "cx"]


def test_openqasm3_declaration_style():
    circuit = parse_qasm("OPENQASM 3; qubit[3] q; bit[3] c; h q[2];")
    assert circuit.num_qubits == 3
    assert circuit.gates[0].qubits == (2,)


def test_comments_and_includes_are_ignored():
    source = """
    // leading comment
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];  // trailing comment
    /* block
       comment */
    h q[0];
    """
    assert len(parse_qasm(source)) == 1


def test_statements_may_span_lines():
    circuit = parse_qasm("qreg q[2];\ncx q[0],\n   q[1];")
    assert circuit.gates[0].qubits == (0, 1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.5", 0.5),
        ("-0.25", -0.25),
        ("pi", math.pi),
        ("-pi", -math.pi),
        ("pi/4", math.pi / 4),
        ("2*pi", 2 * math.pi),
        ("3*pi/8", 3 * math.pi / 8),
        ("1e-3", 1e-3),
    ],
)
def test_parameter_forms(text, expected):
    circuit = parse_qasm(f"qreg q[1]; rz({text}) q[0];")
    assert circuit.gates[0].params[0] == pytest.approx(expected)


def test_measure_arrow_form():
    circuit = parse_qasm("qreg q[2]; creg c[2]; measure q[1] -> c[1];")
    assert circuit.measured_qubits == (1,)


def test_measure_assignment_form():
    circuit = parse_qasm("OPENQASM 3; qubit[2] q; bit[2] c; c[0] = measure q[0];")
    assert circuit.measured_qubits == (0,)


def test_bare_measure_form():
    circuit = parse_qasm("qreg q[2]; measure q[0];")
    assert circuit.measured_qubits == (0,)


def test_barrier_is_accepted_and_ignored():
    circuit = parse_qasm("qreg q[2]; h q[0]; barrier q; h q[1];")
    assert len(circuit) == 2


@pytest.mark.parametrize(
    "source, message",
    [
        ("qreg q[2]; toffoli q[0], q[1];", "outside the AegisQ instruction set"),
        ("h q[0];", "before any quantum register"),
        ("qreg q[2]; h q[7];", "outside the declared register"),
        ("qreg q[2]; rz(alpha) q[0];", "unsupported parameter expression"),
        ("qreg q[2]; h q[0]", "missing a terminating"),
        ("qreg q[2]; rz q[0];", "takes 1 parameter"),
        ("qreg q[2]; h q[0], q[1];", "acts on 1 qubit"),
        ("qreg q[2]; cx q[0], q[0];", "distinct"),
        ("qreg q[2]; qreg r[2];", "single quantum register"),
        ("qreg q[2]; h r[0];", "unknown register"),
        ("qreg q[2]; h q;", "register-wide application is not supported"),
        ("qreg q[2]; reset q[0];", "reset is not supported"),
        ("qreg q[2]; gate mygate a { h a; }", "custom gate definitions"),
        ("h;", "before any quantum register"),
    ],
)
def test_unsupported_constructs_are_rejected_with_context(source, message):
    with pytest.raises(QasmError, match=message):
        parse_qasm(source)


def test_error_messages_carry_a_line_number():
    source = "qreg q[2];\nh q[0];\ntoffoli q[0], q[1];"
    with pytest.raises(QasmError, match="line 3"):
        parse_qasm(source)


def test_no_register_at_all_is_rejected():
    with pytest.raises(QasmError, match="no quantum register"):
        parse_qasm("OPENQASM 2.0;")


def test_roundtrip_through_emitter():
    from aegisq.algorithms import qft

    original = qft(5)
    original.measure_all()
    restored = parse_qasm(to_qasm(original))
    assert [str(g) for g in restored] == [str(g) for g in original]
    assert restored.measured_qubits == original.measured_qubits


def test_roundtrip_preserves_simulation_results():
    from aegisq.algorithms import random_circuit

    original = random_circuit(6, depth=8, seed=4)
    restored = parse_qasm(to_qasm(original, version="3"))
    a = Simulator("cpp").run(original).statevector
    b = Simulator("cpp").run(restored).statevector
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("filename", ["bell.qasm", "ghz8.qasm"])
def test_shipped_examples_parse_and_run(filename):
    circuit = parse_qasm_file(EXAMPLES / filename)
    result = Simulator("cpp").run(circuit, shots=200, seed=1)
    assert sum(result.counts.values()) == 200
    # Both examples prepare a GHZ-type state: only all-zeros and all-ones.
    assert set(result.counts) == {"0" * circuit.num_qubits, "1" * circuit.num_qubits}


def test_emitter_rejects_unknown_version():
    with pytest.raises(ValueError, match="version"):
        to_qasm(parse_qasm("qreg q[1]; h q[0];"), version="4")
