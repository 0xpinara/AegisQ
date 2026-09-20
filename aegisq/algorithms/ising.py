"""Trotterised time evolution of a 1D transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

One Trotter step applies `exp(i J dt Z Z)` on every bond and `exp(i h dt X)`
on every site. The ZZ interaction is built from `cx . rz . cx`, giving a
circuit whose communication is dominated by nearest-neighbour CX pairs — a
realistic middle ground between the GHZ chain and the all-to-all QFT.
"""

from __future__ import annotations

from aegisq.circuit.circuit import Circuit


def zz_interaction(circuit: Circuit, a: int, b: int, theta: float) -> Circuit:
    """Append exp(-i theta Z_a Z_b / 2)."""
    circuit.cx(a, b)
    circuit.rz(b, theta)
    circuit.cx(a, b)
    return circuit


def ising_trotter(
    num_qubits: int,
    steps: int = 4,
    coupling: float = 1.0,
    field: float = 0.8,
    dt: float = 0.1,
    periodic: bool = False,
    name: str = "ising",
) -> Circuit:
    """`steps` first-order Trotter steps of the transverse-field Ising model."""
    circuit = Circuit(num_qubits, name=f"{name}{num_qubits}s{steps}")
    # Start from |+>^n so the dynamics are non-trivial from the first step.
    for q in range(num_qubits):
        circuit.h(q)

    bonds = [(q, q + 1) for q in range(num_qubits - 1)]
    if periodic and num_qubits > 2:
        bonds.append((num_qubits - 1, 0))

    for _ in range(steps):
        for a, b in bonds:
            zz_interaction(circuit, a, b, -2.0 * coupling * dt)
        for q in range(num_qubits):
            circuit.rx(q, -2.0 * field * dt)
    return circuit
