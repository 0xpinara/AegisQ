"""How much does the placement heuristic give up?

The static mapper searches exhaustively while the candidate count fits inside
a budget, and falls back to a greedy start with pairwise local search beyond
it. The result already says which strategy it used and whether it is optimal
with respect to the cost model — but "not guaranteed optimal" is not a
quantity. This measures the quantity.

For sizes where the exhaustive search is affordable, both strategies are run
on the same circuit and the gap is recorded:

    gap = (heuristic bytes - optimal bytes) / optimal bytes

together with the time each search took, because the heuristic exists to buy
time and the trade has to be visible from both sides.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

PLACEMENT_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "os",
    "aegisq_version",
    "git_commit",
    "git_dirty",
    "experiment",
    "circuit_family",
    "seed",
    "qubits",
    "gates",
    "ranks",
    "candidates",
    "fallback_strategy",
    "optimal_bytes",
    "heuristic_bytes",
    "default_bytes",
    "gap",
    "found_optimum",
    "optimal_seconds",
    "heuristic_seconds",
    "speedup",
]


def _environment() -> dict[str, Any]:
    from aegisq.benchmark.runner import provenance_row

    return provenance_row()


def compare_strategies(
    circuit,
    world_size: int,
    precision: str = "fp64",
    family: str = "random",
    seed: int = 0,
) -> dict[str, Any]:
    """Run both searches on one circuit and record the gap between them."""
    from aegisq.compiler import CommunicationCostModel, StaticCommunicationMapper
    from aegisq.compiler.cost_model import default_global_qubits

    model = CommunicationCostModel(circuit.num_qubits, world_size, precision)
    candidates = math.comb(model.num_qubits, model.num_global_qubits)

    exhaustive = StaticCommunicationMapper(model, candidate_budget=candidates + 1)
    heuristic = StaticCommunicationMapper(model, candidate_budget=1)

    optimal = exhaustive.optimize(circuit)
    approximate = heuristic.optimize(circuit)
    # Which fallback applies is itself a result: a swap-free circuit has a
    # separable objective and takes the exact linear path rather than the
    # heuristic, so a zero gap on those rows is expected by construction and
    # the row records which path was taken.
    fallback = approximate.strategy

    optimal_bytes = optimal.optimized.bytes_sent
    heuristic_bytes = approximate.optimized.bytes_sent
    gap = (heuristic_bytes - optimal_bytes) / optimal_bytes if optimal_bytes else 0.0

    return {
        **_environment(),
        "experiment": "placement_quality",
        "circuit_family": family,
        "seed": seed,
        "qubits": circuit.num_qubits,
        "gates": len(circuit),
        "ranks": world_size,
        "candidates": candidates,
        "fallback_strategy": fallback,
        "optimal_bytes": optimal_bytes,
        "heuristic_bytes": heuristic_bytes,
        "default_bytes": model.estimate(
            circuit, default_global_qubits(model.num_qubits, world_size)
        ).bytes_sent,
        "gap": gap,
        "found_optimum": int(heuristic_bytes == optimal_bytes),
        "optimal_seconds": optimal.search_seconds,
        "heuristic_seconds": approximate.search_seconds,
        "speedup": (
            optimal.search_seconds / approximate.search_seconds
            if approximate.search_seconds > 0
            else 0.0
        ),
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLACEMENT_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PLACEMENT_RAW_FIELDS})


def run_suite(
    output: Path,
    qubits: int = 18,
    ranks: list[int] | None = None,
    samples: int = 40,
    families: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Sample circuits and compare the two searches on each."""
    from aegisq.algorithms import build_circuit, random_circuit

    ranks = ranks or [4, 8]
    families = families or ["random", "qft", "ising", "grover"]
    rows: list[dict[str, Any]] = []

    for world_size in ranks:
        for family in families:
            count = samples if family == "random" else 1
            for seed in range(count):
                if family == "random":
                    circuit = random_circuit(qubits, depth=12, seed=seed)
                else:
                    circuit = build_circuit(
                        family, qubits, **({"iterations": 2} if family == "grover" else {})
                    )
                rows.append(compare_strategies(circuit, world_size, family=family, seed=seed))
            if verbose:
                subset = [
                    r for r in rows if r["ranks"] == world_size and r["circuit_family"] == family
                ]
                found = sum(r["found_optimum"] for r in subset)
                worst = max(r["gap"] for r in subset)
                print(
                    f"  {family:7s} {world_size} ranks: optimum found in "
                    f"{found}/{len(subset)} cases, worst gap {worst * 100:.1f}%"
                )

    append_rows(output, rows)
    return output
