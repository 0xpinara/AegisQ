"""Deterministic pseudo-random circuits.

Used as a benchmark family and as the workload for cross-backend validation.
The generator is seeded and depends on nothing but its arguments, so a failing
case is always reproducible from the seed printed in the test output.
"""

from __future__ import annotations

import random

from aegisq.circuit.circuit import Circuit

_ONE_QUBIT = ("x", "y", "z", "h", "s", "t")
_ROTATIONS = ("rx", "ry", "rz")
_TWO_QUBIT = ("cx", "cz", "swap")


def random_circuit(
    num_qubits: int,
    depth: int,
    seed: int,
    *,
    two_qubit_prob: float = 0.35,
    name: str = "random",
) -> Circuit:
    """A layered random circuit over the supported gate set.

    Each layer applies a random single-qubit gate to roughly three quarters of
    the qubits, then pairs the qubits up at random and applies a two-qubit
    gate to each pair with probability `two_qubit_prob * 2`.
    """
    rng = random.Random(seed)
    circuit = Circuit(num_qubits, name=f"{name}_n{num_qubits}_d{depth}_s{seed}")

    for _ in range(depth):
        for q in range(num_qubits):
            roll = rng.random()
            if roll < 0.45:
                getattr(circuit, rng.choice(_ONE_QUBIT))(q)
            elif roll < 0.75:
                getattr(circuit, rng.choice(_ROTATIONS))(q, rng.uniform(-3.14, 3.14))
        if num_qubits >= 2:
            order = list(range(num_qubits))
            rng.shuffle(order)
            for a, b in zip(order[::2], order[1::2], strict=False):
                if rng.random() < two_qubit_prob * 2:
                    getattr(circuit, rng.choice(_TWO_QUBIT))(a, b)
    return circuit
