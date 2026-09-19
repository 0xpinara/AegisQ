"""Gate definitions for the AegisQ instruction set.

The supported set is deliberately small. Every gate here must be implementable
by three independent backends — the NumPy reference simulator, the C++
single-process engine, and the distributed MPI runtime — so each addition has
a real cost. Higher-level operations (controlled phase, QFT, Grover oracles)
are decomposed into this set by ``aegisq.algorithms``.

Two structural properties are recorded per gate because the distributed
runtime and the communication cost model depend on them:

``diagonal``
    The matrix is diagonal in the computational basis. A diagonal gate never
    moves an amplitude between basis states, so it needs no MPI communication
    regardless of whether the qubit is local or global.
``control_positions``
    Indices into ``Gate.qubits`` that act as controls. A control qubit is only
    *read*; when it lives on a global (rank-selecting) position its value is
    determined by the rank id, which again avoids communication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Amplitude dtype used by the reference backend.
COMPLEX = np.complex128

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class GateSpec:
    """Static description of one supported operation."""

    opcode: str
    num_qubits: int
    num_params: int
    diagonal: bool
    control_positions: tuple[int, ...]
    description: str

    @property
    def is_two_qubit(self) -> bool:
        return self.num_qubits == 2


GATE_SPECS: dict[str, GateSpec] = {
    "x": GateSpec("x", 1, 0, False, (), "Pauli-X (bit flip)"),
    "y": GateSpec("y", 1, 0, False, (), "Pauli-Y"),
    "z": GateSpec("z", 1, 0, True, (), "Pauli-Z (phase flip)"),
    "h": GateSpec("h", 1, 0, False, (), "Hadamard"),
    "s": GateSpec("s", 1, 0, True, (), "Phase gate, sqrt(Z)"),
    "t": GateSpec("t", 1, 0, True, (), "T gate, fourth root of Z"),
    "rx": GateSpec("rx", 1, 1, False, (), "Rotation about X"),
    "ry": GateSpec("ry", 1, 1, False, (), "Rotation about Y"),
    "rz": GateSpec("rz", 1, 1, True, (), "Rotation about Z"),
    "cx": GateSpec("cx", 2, 0, False, (0,), "Controlled-X (control, target)"),
    "cz": GateSpec("cz", 2, 0, True, (0,), "Controlled-Z (symmetric)"),
    "swap": GateSpec("swap", 2, 0, False, (), "Exchange two qubits"),
}

#: Opcodes accepted anywhere in AegisQ, in a stable documentation order.
SUPPORTED_OPCODES: tuple[str, ...] = tuple(GATE_SPECS)


@dataclass(frozen=True)
class Gate:
    """One instruction: an opcode, its qubit operands and its parameters."""

    opcode: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    @property
    def spec(self) -> GateSpec:
        return GATE_SPECS[self.opcode]

    @property
    def is_diagonal(self) -> bool:
        return self.spec.diagonal

    @property
    def control_qubits(self) -> tuple[int, ...]:
        return tuple(self.qubits[i] for i in self.spec.control_positions)

    @property
    def target_qubits(self) -> tuple[int, ...]:
        controls = set(self.spec.control_positions)
        return tuple(q for i, q in enumerate(self.qubits) if i not in controls)

    def __str__(self) -> str:
        # Stays printable for not-yet-validated gates, because validation
        # errors embed the offending instruction in their message.
        args = ", ".join(f"q{q}" for q in self.qubits)
        if self.params:
            params = ", ".join(
                f"{p:.6g}" if isinstance(p, (int, float)) and not isinstance(p, bool) else repr(p)
                for p in self.params
            )
            return f"{self.opcode}({params}) {args}"
        return f"{self.opcode} {args}"


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=COMPLEX)
_Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX)
_Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX)
_H = np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=COMPLEX)
_S = np.array([[1, 0], [0, 1j]], dtype=COMPLEX)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=COMPLEX)


def rx_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=COMPLEX)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=COMPLEX)


def rz_matrix(theta: float) -> np.ndarray:
    phase = np.exp(-1j * theta / 2.0)
    return np.array([[phase, 0], [0, np.conjugate(phase)]], dtype=COMPLEX)


_STATIC_MATRICES: dict[str, np.ndarray] = {
    "x": _X,
    "y": _Y,
    "z": _Z,
    "h": _H,
    "s": _S,
    "t": _T,
}

_PARAMETRIC_MATRICES = {
    "rx": rx_matrix,
    "ry": ry_matrix,
    "rz": rz_matrix,
}


def single_qubit_matrix(gate: Gate) -> np.ndarray:
    """2x2 unitary for a one-qubit gate."""
    if gate.spec.num_qubits != 1:
        raise ValueError(f"{gate.opcode} is not a single-qubit gate")
    if gate.opcode in _STATIC_MATRICES:
        return _STATIC_MATRICES[gate.opcode]
    return _PARAMETRIC_MATRICES[gate.opcode](gate.params[0])


def two_qubit_matrix(gate: Gate) -> np.ndarray:
    """4x4 unitary for a two-qubit gate, in ``|q0 q1>`` operand order.

    Row/column index ``2*a + b`` corresponds to ``qubits[0] = a`` and
    ``qubits[1] = b``; that is, the first operand is the more significant bit
    of the local 2-qubit index.
    """
    if gate.spec.num_qubits != 2:
        raise ValueError(f"{gate.opcode} is not a two-qubit gate")
    if gate.opcode == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=COMPLEX)
    if gate.opcode == "cz":
        return np.diag(np.array([1, 1, 1, -1], dtype=COMPLEX))
    if gate.opcode == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=COMPLEX)
    raise ValueError(f"no matrix defined for {gate.opcode}")


def inverse(gate: Gate) -> Gate:
    """Return the inverse instruction.

    ``S`` and ``T`` have no dagger form in the supported gate set, so their
    inverses are expressed as ``RZ`` rotations. Those reproduce the adjoint up
    to a global phase (``S @ RZ(-pi/2) = e^{i pi/4} I``), which is unobservable
    for state preparation and measurement. It would matter if these
    instructions were ever promoted to controlled operations, so the
    substitution is recorded here rather than hidden.
    """
    if gate.opcode in ("rx", "ry", "rz"):
        return Gate(gate.opcode, gate.qubits, (-gate.params[0],))
    if gate.opcode in ("x", "y", "z", "h", "cx", "cz", "swap"):
        return gate  # self-inverse
    if gate.opcode == "s":
        return Gate("rz", gate.qubits, (-math.pi / 2,))
    if gate.opcode == "t":
        return Gate("rz", gate.qubits, (-math.pi / 4,))
    raise ValueError(f"no inverse defined for {gate.opcode}")
