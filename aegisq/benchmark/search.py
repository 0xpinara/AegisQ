"""Measure Grover's query scaling.

Research question 3 asks how a small Grover experiment illustrates the
cryptographic motivation for post-quantum cryptography. The illustration is
about **oracle queries**, not wall time:

* a classical search of an unstructured space of `N` items needs `(N+1)/2`
  oracle queries on average and `N` in the worst case;
* Grover needs `floor(pi/4 * sqrt(N))` — a quadratic reduction.

That quadratic factor is exactly why symmetric key sizes are doubled in
post-quantum guidance while public-key schemes are replaced outright: Grover
halves the effective bits of a symmetric key, whereas Shor breaks RSA and
elliptic curves entirely.

This module runs the circuits and records the **measured** success
probability alongside the query counts, so the plot shows what the simulation
actually did rather than the textbook formula alone.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

SEARCH_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "os",
    "aegisq_version",
    "git_commit",
    "git_dirty",
    "experiment",
    "search_bits",
    "search_space",
    "marked_state",
    "qubits",
    "gates",
    "depth",
    "grover_iterations",
    "classical_expected_queries",
    "classical_worst_case_queries",
    "measured_success_probability",
    "theoretical_success_probability",
    "shots",
    "seed",
    "wall_seconds",
]


def theoretical_success(search_bits: int, iterations: int) -> float:
    """Amplitude-amplification success probability after `iterations` steps."""
    import math

    space = 2**search_bits
    theta = math.asin(1 / math.sqrt(space))
    return float(math.sin((2 * iterations + 1) * theta) ** 2)


def measure_grover(
    search_bits: int,
    marked: int | None = None,
    shots: int = 2048,
    seed: int = 42,
    backend: str = "cpp",
) -> dict[str, Any]:
    """Run one Grover search and record queries and measured success."""

    from aegisq.algorithms.grover import grover, optimal_iterations
    from aegisq.benchmark.runner import provenance_row
    from aegisq.runtime import Simulator

    space = 2**search_bits
    marked = (space - 5) % space if marked is None else marked % space
    iterations = optimal_iterations(search_bits)
    circuit = grover(search_bits, marked=marked, iterations=iterations)

    started = time.perf_counter()
    result = Simulator(backend).run(circuit, shots=shots, seed=seed)
    wall = time.perf_counter() - started

    key = format(marked, f"0{search_bits}b")
    hits = result.counts.get(key, 0)

    return {
        **provenance_row(),
        "experiment": "grover_scaling",
        "search_bits": search_bits,
        "search_space": space,
        "marked_state": marked,
        "qubits": circuit.num_qubits,
        "gates": len(circuit),
        "depth": circuit.depth(),
        "grover_iterations": iterations,
        "classical_expected_queries": (space + 1) / 2,
        "classical_worst_case_queries": space,
        "measured_success_probability": hits / shots,
        "theoretical_success_probability": theoretical_success(search_bits, iterations),
        "shots": shots,
        "seed": seed,
        "wall_seconds": wall,
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCH_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SEARCH_RAW_FIELDS})


def run_suite(
    output: Path,
    bits: list[int] | None = None,
    shots: int = 2048,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Sweep search-space sizes and record one row each."""
    rows = []
    for search_bits in bits or list(range(2, 9)):
        if verbose:
            print(f"  searching 2^{search_bits} = {2**search_bits} items")
        row = measure_grover(search_bits, shots=shots, seed=seed)
        rows.append(row)
        if verbose:
            print(
                f"    {row['grover_iterations']} oracle queries vs "
                f"{row['classical_expected_queries']:.1f} classical, "
                f"success {row['measured_success_probability'] * 100:.1f}%"
            )
    append_rows(output, rows)
    return output
