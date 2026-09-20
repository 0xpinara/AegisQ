"""GHZ state preparation.

The opposite extreme from the QFT: a single Hadamard followed by a CX chain,
so the communication pattern is a thin line rather than an all-to-all mesh.
Including both in the benchmark suite keeps the mapper honest — a placement
heuristic that only helps dense circuits is not much of a result.
"""

from __future__ import annotations

from aegisq.circuit.circuit import Circuit


def ghz(num_qubits: int, name: str = "ghz") -> Circuit:
    """(|0...0> + |1...1>) / sqrt(2) via a nearest-neighbour CX chain."""
    circuit = Circuit(num_qubits, name=f"{name}{num_qubits}")
    circuit.h(0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    return circuit


def ghz_layered(num_qubits: int, repeats: int = 1, name: str = "ghz-layered") -> Circuit:
    """GHZ preparation repeated (and undone) to give the benchmark some depth."""
    circuit = Circuit(num_qubits, name=f"{name}{num_qubits}x{repeats}")
    for r in range(repeats):
        circuit.h(0)
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)
        if r + 1 < repeats:
            for q in reversed(range(num_qubits - 1)):
                circuit.cx(q, q + 1)
            circuit.h(0)
    return circuit
