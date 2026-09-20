"""Quantum Fourier transform, decomposed into the AegisQ gate set.

The QFT is the canonical communication-heavy benchmark: every qubit pair is
connected by a controlled phase, so a distributed run touches every rank
boundary. AegisQ has no controlled-phase primitive, so `CP(theta)` is built
from `rz` and `cx`:

    CP(theta) = RZ_c(theta/2) RZ_t(theta/2) CX RZ_t(-theta/2) CX

which reproduces the controlled phase up to a global phase (the exact identity
uses the phase gate `P`; `P(l) = exp(i l / 2) RZ(l)`). Global phase is
unobservable, and the Qiskit cross-validation suite checks this construction
explicitly.
"""

from __future__ import annotations

import math

from aegisq.circuit.circuit import Circuit


def controlled_phase(circuit: Circuit, control: int, target: int, theta: float) -> Circuit:
    """Append a controlled phase rotation, up to a global phase."""
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cx(control, target)
    circuit.rz(target, -theta / 2)
    circuit.cx(control, target)
    return circuit


def qft(num_qubits: int, swaps: bool = True, name: str = "qft") -> Circuit:
    """Textbook QFT over `num_qubits` qubits.

    `swaps` appends the final bit-reversal. It is often skipped in practice
    (the reversal can be absorbed into the surrounding algorithm), and it is
    worth keeping optional here because those swaps are themselves expensive
    in a distributed run.
    """
    circuit = Circuit(num_qubits, name=f"{name}{num_qubits}")
    # Qubit 0 is the least significant bit, so the transform starts at the
    # most significant qubit and its controls are the *less* significant ones.
    # Running the loop the other way round produces a perfectly reasonable
    # circuit that is not the Fourier transform; `tests/unit/test_algorithms.py`
    # pins this by comparing the whole matrix against the DFT.
    for j in reversed(range(num_qubits)):
        circuit.h(j)
        for k in range(j):
            controlled_phase(circuit, k, j, math.pi / (2 ** (j - k)))
    if swaps:
        for j in range(num_qubits // 2):
            circuit.swap(j, num_qubits - 1 - j)
    return circuit


def inverse_qft(num_qubits: int, swaps: bool = True, name: str = "iqft") -> Circuit:
    """Inverse QFT, used by the phase-estimation step of Shor's algorithm."""
    forward = qft(num_qubits, swaps=swaps)
    inverted = forward.inverse()
    inverted.name = f"{name}{num_qubits}"
    return inverted
