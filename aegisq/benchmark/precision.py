"""What does single precision buy, and what does it cost?

fp32 halves everything the simulator is bound by: half the memory per
amplitude, half the bytes on the wire, and — in principle — twice the
bandwidth. Unlike placement or fusion, the saving is exact and needs no
analysis: the shard is half the size, so every transfer is half the bytes.

The interesting quantity is therefore not the saving but the error, and how
it grows. Rounding accumulates with circuit depth, so "fp32 is fine" is a
statement about a depth, not about a precision. This module measures both
sides on the same circuits:

* traffic and wall time at each precision, from the runtime's own counters;
* the fidelity of the fp32 state against the fp64 one, and the largest
  amplitude discrepancy, as a function of depth.

Fidelity is `|<psi32|psi64>|^2`, which is the quantity that governs how
different the two would look to any measurement.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

PRECISION_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "aegisq_version",
    "git_commit",
    "experiment",
    "circuit_family",
    "qubits",
    "depth",
    "gates",
    "two_qubit_gates",
    "seed",
    "fidelity",
    "infidelity",
    "max_amplitude_error",
    "norm_error",
    "fp64_seconds",
    "fp32_seconds",
    "speedup",
]


def _environment() -> dict[str, Any]:
    import socket

    from aegisq import __version__
    from aegisq.benchmark.runner import git_commit
    from aegisq.runtime import hardware

    commit, _ = git_commit()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "cpu_model": hardware.cpu_info().extra.get("model", "unknown"),
        "aegisq_version": __version__,
        "git_commit": commit,
    }


def compare_precisions(circuit, family: str = "random", seed: int = 0) -> dict[str, Any]:
    """Run one circuit at both precisions and measure the difference."""
    import numpy as np

    from aegisq.runtime import Simulator

    started = time.perf_counter()
    double = Simulator("cpp", precision="fp64").run(circuit).statevector
    fp64_seconds = time.perf_counter() - started

    started = time.perf_counter()
    single = Simulator("cpp", precision="fp32").run(circuit).statevector
    fp32_seconds = time.perf_counter() - started

    double = np.asarray(double, dtype=np.complex128)
    single = np.asarray(single, dtype=np.complex128)

    overlap = np.vdot(double, single)
    fidelity = float(
        abs(overlap) ** 2 / (np.vdot(double, double).real * np.vdot(single, single).real)
    )

    return {
        **_environment(),
        "experiment": "precision",
        "circuit_family": family,
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gates": len(circuit),
        "two_qubit_gates": circuit.two_qubit_gate_count(),
        "seed": seed,
        "fidelity": fidelity,
        # Stored explicitly: 1 - fidelity is the quantity that actually
        # varies, and it disappears into rounding if a reader has to compute
        # it from a fidelity printed to a handful of decimals.
        "infidelity": 1.0 - fidelity,
        "max_amplitude_error": float(np.abs(double - single).max()),
        "norm_error": float(abs(1.0 - np.vdot(single, single).real)),
        "fp64_seconds": fp64_seconds,
        "fp32_seconds": fp32_seconds,
        "speedup": fp64_seconds / fp32_seconds if fp32_seconds > 0 else 0.0,
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRECISION_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PRECISION_RAW_FIELDS})


def run_suite(
    output: Path,
    qubits: int = 18,
    depths: list[int] | None = None,
    families: list[str] | None = None,
    samples: int = 5,
    verbose: bool = True,
) -> Path:
    """Sweep circuit depth, which is what governs how far fp32 drifts."""
    from aegisq.algorithms import build_circuit, random_circuit

    depths = depths or [2, 4, 8, 16, 32, 64]
    families = families or ["random", "qft", "ising", "grover"]
    rows: list[dict[str, Any]] = []

    for depth in depths:
        for seed in range(samples):
            rows.append(
                compare_precisions(random_circuit(qubits, depth=depth, seed=seed), "random", seed)
            )
        if verbose:
            recent = [r for r in rows if r["circuit_family"] == "random" and r["depth"] > 0][
                -samples:
            ]
            worst = min(r["fidelity"] for r in recent)
            print(
                f"  random, {depth:3d} layers ({recent[0]['gates']:5d} gates): "
                f"worst infidelity {1 - worst:.3e}, "
                f"max amplitude error {max(r['max_amplitude_error'] for r in recent):.3e}"
            )

    # The structured families have a fixed shape, so they are measured once
    # each rather than swept.
    for family in families:
        if family == "random":
            continue
        circuit = build_circuit(family, qubits, **({"iterations": 2} if family == "grover" else {}))
        row = compare_precisions(circuit, family, 0)
        rows.append(row)
        if verbose:
            print(
                f"  {family:7s} ({row['gates']:5d} gates, depth {row['depth']:4d}): "
                f"infidelity {1 - row['fidelity']:.3e}, "
                f"max amplitude error {row['max_amplitude_error']:.3e}"
            )

    append_rows(output, rows)
    return output
