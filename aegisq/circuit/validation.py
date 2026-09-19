"""Validation rules shared by the circuit builder, the QASM parser and the runtime.

Errors are raised eagerly and with the offending instruction in the message:
a malformed circuit that reaches the distributed runtime is far more expensive
to diagnose than one rejected at construction time.
"""

from __future__ import annotations

from aegisq.circuit.gates import GATE_SPECS, SUPPORTED_OPCODES, Gate


class CircuitError(ValueError):
    """Raised when a circuit or instruction violates the AegisQ gate contract."""


def validate_num_qubits(num_qubits: int) -> int:
    if not isinstance(num_qubits, int) or isinstance(num_qubits, bool):
        raise CircuitError(f"qubit count must be an int, got {type(num_qubits).__name__}")
    if num_qubits < 1:
        raise CircuitError(f"a circuit needs at least one qubit, got {num_qubits}")
    return num_qubits


def validate_opcode(opcode: str) -> str:
    key = opcode.lower()
    if key not in GATE_SPECS:
        raise CircuitError(
            f"unsupported operation '{opcode}'; AegisQ supports {', '.join(SUPPORTED_OPCODES)}"
        )
    return key


def validate_gate(gate: Gate, num_qubits: int) -> Gate:
    """Check opcode, arity, parameter count and operand indices."""
    opcode = validate_opcode(gate.opcode)
    spec = GATE_SPECS[opcode]

    if len(gate.qubits) != spec.num_qubits:
        raise CircuitError(
            f"{opcode} acts on {spec.num_qubits} qubit(s), got {len(gate.qubits)}: {gate}"
        )
    if len(gate.params) != spec.num_params:
        raise CircuitError(
            f"{opcode} takes {spec.num_params} parameter(s), got {len(gate.params)}: {gate}"
        )
    for q in gate.qubits:
        if not isinstance(q, int) or isinstance(q, bool):
            raise CircuitError(f"qubit index must be an int, got {q!r} in {gate}")
        if not 0 <= q < num_qubits:
            raise CircuitError(
                f"qubit index {q} out of range for a {num_qubits}-qubit circuit: {gate}"
            )
    if len(set(gate.qubits)) != len(gate.qubits):
        raise CircuitError(f"{opcode} requires distinct qubits: {gate}")
    for p in gate.params:
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise CircuitError(f"gate parameter must be a real number, got {p!r} in {gate}")

    return Gate(opcode, tuple(int(q) for q in gate.qubits), tuple(float(p) for p in gate.params))
