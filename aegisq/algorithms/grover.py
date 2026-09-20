"""Grover search on educational problem sizes.

This is a demonstration of the *query* scaling that motivates post-quantum
cryptography, not a claim about breaking anything. The search spaces here are
`2^k` for small `k`; the classical baseline is `O(N)` oracle queries and
Grover needs `O(sqrt(N))`.

AegisQ's gate set has no Toffoli, so multi-controlled operations are built
from the standard 6-CX decomposition with a ladder of ancilla qubits. `T-dagger`
is expressed as `rz(-pi/4)`, which differs from the exact adjoint by a global
phase; since these are unconditional single-qubit gates the phase multiplies
the whole state and is unobservable.
"""

from __future__ import annotations

import math

from aegisq.circuit.circuit import Circuit


def toffoli(circuit: Circuit, a: int, b: int, target: int) -> Circuit:
    """CCX via the textbook 6-CX / 7-T decomposition."""
    quarter = math.pi / 4
    circuit.h(target)
    circuit.cx(b, target)
    circuit.rz(target, -quarter)  # T-dagger up to a global phase
    circuit.cx(a, target)
    circuit.t(target)
    circuit.cx(b, target)
    circuit.rz(target, -quarter)
    circuit.cx(a, target)
    circuit.t(b)
    circuit.t(target)
    circuit.h(target)
    circuit.cx(a, b)
    circuit.t(a)
    circuit.rz(b, -quarter)
    circuit.cx(a, b)
    return circuit


def multi_controlled_z(
    circuit: Circuit, controls: list[int], target: int, ancillas: list[int]
) -> Circuit:
    """Apply Z to `target` conditioned on every qubit in `controls`.

    Uses an ancilla ladder: `ancillas[i]` holds the AND of everything up to
    control `i + 2`. The ladder is uncomputed afterwards, so the ancillas
    return to |0> and stay unentangled.
    """
    if not controls:
        circuit.z(target)
        return circuit
    if len(controls) == 1:
        circuit.cz(controls[0], target)
        return circuit

    needed = len(controls) - 1
    if len(ancillas) < needed:
        raise ValueError(
            f"multi-controlled Z over {len(controls)} controls needs {needed} ancillas"
        )

    toffoli(circuit, controls[0], controls[1], ancillas[0])
    for i in range(2, len(controls)):
        toffoli(circuit, controls[i], ancillas[i - 2], ancillas[i - 1])

    circuit.cz(ancillas[needed - 1], target)

    for i in reversed(range(2, len(controls))):
        toffoli(circuit, controls[i], ancillas[i - 2], ancillas[i - 1])
    toffoli(circuit, controls[0], controls[1], ancillas[0])
    return circuit


def grover_oracle(circuit: Circuit, marked: int, search_qubits: list[int], ancillas: list[int]):
    """Flip the phase of the single marked basis state."""
    for index, q in enumerate(search_qubits):
        if not (marked >> index) & 1:
            circuit.x(q)
    multi_controlled_z(circuit, search_qubits[:-1], search_qubits[-1], ancillas)
    for index, q in enumerate(search_qubits):
        if not (marked >> index) & 1:
            circuit.x(q)
    return circuit


def grover_diffusion(circuit: Circuit, search_qubits: list[int], ancillas: list[int]):
    """Inversion about the mean."""
    for q in search_qubits:
        circuit.h(q)
        circuit.x(q)
    multi_controlled_z(circuit, search_qubits[:-1], search_qubits[-1], ancillas)
    for q in search_qubits:
        circuit.x(q)
        circuit.h(q)
    return circuit


def optimal_iterations(search_bits: int) -> int:
    """Iterations that maximise the marked amplitude for one marked item.

    `floor(pi/4 * sqrt(N))` — the sqrt(N) scaling this demonstration exists to
    illustrate.
    """
    n = 2**search_bits
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(n))))


def grover(
    search_bits: int,
    marked: int = 0b1011,
    iterations: int | None = None,
    name: str = "grover",
) -> Circuit:
    """Grover search over `2^search_bits` items with one marked item.

    The circuit uses `search_bits + max(0, search_bits - 2)` qubits: the
    search register plus the ancilla ladder for the multi-controlled phase
    flip.
    """
    if search_bits < 2:
        raise ValueError("Grover needs at least two search qubits")
    marked %= 2**search_bits

    num_ancillas = max(0, search_bits - 2)
    total = search_bits + num_ancillas
    search_qubits = list(range(search_bits))
    ancillas = list(range(search_bits, total))

    if iterations is None:
        iterations = optimal_iterations(search_bits)

    circuit = Circuit(total, name=f"{name}{search_bits}x{iterations}")
    for q in search_qubits:
        circuit.h(q)
    for _ in range(iterations):
        grover_oracle(circuit, marked, search_qubits, ancillas)
        grover_diffusion(circuit, search_qubits, ancillas)
    for q in search_qubits:
        circuit.measure(q)
    return circuit
