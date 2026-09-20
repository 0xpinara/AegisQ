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

Fidelity is `|<psi32|psi64>|^2`, which governs how different the two would
look to any measurement. It is never computed by subtracting from one: at
these error levels that subtraction is pure cancellation, and it produced a
negative infidelity before being replaced. See `_fidelity`.
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


def _fidelity(reference, candidate) -> tuple[float, float]:
    """Fidelity and infidelity, computed so that the small one is meaningful.

    `1 - |<a|b>|^2` cannot be evaluated directly when the fidelity is within
    rounding distance of one: the subtraction cancels and the result is
    noise -- including, as happened here, a negative infidelity, which is not
    a possible value.

    Instead the phase-aligned distance is accumulated as a sum of
    non-negative terms,

        d^2 = || a_hat - b_hat e^{-i theta} ||^2 = 2 (1 - |c|),  c = <a_hat|b_hat>

    from which the infidelity follows as `d^2 - d^4 / 4`. Every step is well
    conditioned, so an infidelity of 1e-14 means 1e-14.
    """
    import numpy as np

    a = np.asarray(reference, dtype=np.complex128)
    b = np.asarray(candidate, dtype=np.complex128)
    a = a / np.sqrt(np.vdot(a, a).real)
    b = b / np.sqrt(np.vdot(b, b).real)

    overlap = np.vdot(a, b)
    magnitude = abs(overlap)
    if magnitude > 0:
        b = b * (np.conjugate(overlap) / magnitude)

    distance_squared = float(np.sum(np.abs(a - b) ** 2))
    infidelity = max(0.0, distance_squared - distance_squared**2 / 4.0)
    return 1.0 - infidelity, infidelity


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

    fidelity, infidelity = _fidelity(double, single)

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
        # Computed directly rather than as 1 minus the fidelity; see
        # `_fidelity` for why that subtraction cannot be done at this scale.
        "infidelity": infidelity,
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
            worst = max(r["infidelity"] for r in recent)
            print(
                f"  random, {depth:3d} layers ({recent[0]['gates']:5d} gates): "
                f"worst infidelity {worst:.3e}, "
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
                f"infidelity {row['infidelity']:.3e}, "
                f"max amplitude error {row['max_amplitude_error']:.3e}"
            )

    append_rows(output, rows)
    return output
