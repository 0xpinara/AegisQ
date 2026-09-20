"""Benchmark circuit families and algorithm demonstrations.

Each family stresses a different communication pattern, which is the point:
a placement heuristic that only helps one shape of circuit is not a result.

| family | pattern |
|---|---|
| `ghz` | a single nearest-neighbour CX chain |
| `qft` | all-to-all controlled phases |
| `ising` | repeated nearest-neighbour ZZ layers |
| `grover` | dense multi-controlled blocks over an ancilla ladder |
| `random` | uniform mixture of the whole gate set |
"""

from __future__ import annotations

from collections.abc import Callable

from aegisq.algorithms.ghz import ghz, ghz_layered
from aegisq.algorithms.grover import grover, optimal_iterations
from aegisq.algorithms.ising import ising_trotter
from aegisq.algorithms.qft import controlled_phase, inverse_qft, qft
from aegisq.algorithms.random_circuit import random_circuit
from aegisq.circuit.circuit import Circuit

__all__ = [
    "ghz",
    "ghz_layered",
    "qft",
    "inverse_qft",
    "controlled_phase",
    "ising_trotter",
    "grover",
    "optimal_iterations",
    "random_circuit",
    "build_circuit",
    "CIRCUIT_FAMILIES",
]


def _build_ghz(num_qubits: int, **options) -> Circuit:
    repeats = int(options.get("repeats", 1))
    return ghz(num_qubits) if repeats == 1 else ghz_layered(num_qubits, repeats)


def _build_qft(num_qubits: int, **options) -> Circuit:
    return qft(num_qubits, swaps=bool(options.get("swaps", True)))


def _build_ising(num_qubits: int, **options) -> Circuit:
    return ising_trotter(
        num_qubits,
        steps=int(options.get("steps", 4)),
        periodic=bool(options.get("periodic", False)),
    )


def _build_random(num_qubits: int, **options) -> Circuit:
    return random_circuit(
        num_qubits,
        depth=int(options.get("depth", 12)),
        seed=int(options.get("seed", 0)),
    )


def _build_grover(num_qubits: int, **options) -> Circuit:
    """Grover sized so the *total* qubit count matches `num_qubits`.

    Grover needs `k` search qubits plus `k - 2` ancillas, so a requested width
    of `n` gives `k = (n + 2) // 2`.
    """
    search_bits = max(2, (num_qubits + 2) // 2)
    circuit = grover(
        search_bits,
        marked=int(options.get("marked", 0b1011)),
        iterations=options.get("iterations"),
    )
    return circuit


#: family name -> builder taking (num_qubits, **options)
CIRCUIT_FAMILIES: dict[str, Callable[..., Circuit]] = {
    "ghz": _build_ghz,
    "qft": _build_qft,
    "ising": _build_ising,
    "grover": _build_grover,
    "random": _build_random,
}


def build_circuit(family: str, num_qubits: int, **options) -> Circuit:
    """Construct a benchmark circuit by family name.

    Grover chooses its own width from `num_qubits` (search register plus
    ancilla ladder), so the returned circuit may be one qubit narrower than
    requested; callers should read `circuit.num_qubits` rather than assume.
    """
    try:
        builder = CIRCUIT_FAMILIES[family]
    except KeyError:
        raise ValueError(
            f"unknown circuit family {family!r}; available: {', '.join(sorted(CIRCUIT_FAMILIES))}"
        ) from None
    return builder(num_qubits, **options)
