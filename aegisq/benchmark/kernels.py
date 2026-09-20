"""How close are the local kernels to what the machine can move?

State-vector simulation is bandwidth-bound: a single-qubit gate reads and
writes every amplitude once and performs a handful of floating-point
operations per element. Wall time on its own says nothing about whether a
kernel is good; it has to be compared against the bandwidth the same machine
achieves on a trivial loop.

The reference is measured the same way as the kernels: same data type, same
compiler flags, same threading. Three shapes are recorded, and the one that
matters is the **in-place scale**, because a gate kernel reads and writes a
single array. A two-array copy is a different traffic shape --- half the TLB
pressure is not comparable --- and a kernel can legitimately beat it, which
makes it a misleading yardstick.

Two dimensions are swept:

* **threads**, which shows where the kernels stop scaling;
* **target qubit index**, which matters because the pairs a gate combines are
  `2^q` apart. For a low-index qubit they are adjacent and the sweep is
  effectively sequential; for a high-index qubit they are half a state vector
  apart, and the access pattern turns into two distant streams.

The second is not an implementation detail: it says that *which* qubit a gate
acts on changes its local cost, a term the communication cost model does not
currently carry.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

#: Bytes moved per gate, as a multiple of the amplitudes the kernel touches.
#: Every touched amplitude is read once and written once.
_TOUCHED_FRACTION = {
    "h": 1.0,  # general single-qubit: the whole state
    "rz": 1.0,  # diagonal: the whole state
    "cx": 0.5,  # controlled: half, where the control bit is set
    "cz": 0.25,  # diagonal and controlled: the |11> quarter
    "swap": 0.5,  # the two mixed quarters
}

KERNEL_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "logical_cores",
    "os",
    "compiler",
    "aegisq_version",
    "git_commit",
    "git_dirty",
    "experiment",
    "kernel",
    "qubits",
    "target_qubit",
    "threads",
    "repeats",
    "seconds_per_gate",
    "bytes_per_gate",
    "gb_per_second",
    "inplace_gb_per_second",
    "triad_gb_per_second",
    "copy_gb_per_second",
    "fraction_of_inplace",
]


def _environment() -> dict[str, Any]:
    from aegisq import native_core
    from aegisq.benchmark.runner import provenance_row

    core = native_core()
    # A bandwidth number is a statement about a compiler as much as a CPU.
    return {
        **provenance_row(),
        "compiler": core.compiler() if core is not None else "none",
    }


def _apply_repeatedly(state, kernel: str, target: int, repeats: int) -> float:
    """Best time for one application of `kernel`, in seconds."""
    partner = 0 if target != 0 else 1

    def once() -> None:
        if kernel == "h":
            state.apply_h(target)
        elif kernel == "rz":
            state.apply_rz(target, 0.37)
        elif kernel == "cx":
            state.apply_cnot(partner, target)
        elif kernel == "cz":
            state.apply_cz(partner, target)
        elif kernel == "swap":
            state.apply_swap(partner, target)
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unknown kernel {kernel!r}")

    once()  # warm up: first touch pays for page faults, not for the kernel
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        once()
        best = min(best, time.perf_counter() - started)
    return best


def measure_kernel(
    kernel: str,
    qubits: int,
    target_qubit: int,
    threads: int,
    repeats: int = 7,
    reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Time one kernel and express it as achieved bandwidth."""
    from aegisq import native_core
    from aegisq.runtime.native import new_state

    core = native_core()
    in_force = core.set_num_threads(threads)

    state = new_state(qubits, "fp64")
    seconds = _apply_repeatedly(state, kernel, target_qubit, repeats)

    amplitudes = (1 << qubits) * _TOUCHED_FRACTION[kernel]
    bytes_moved = int(amplitudes * 16 * 2)  # read and write, complex128

    if reference is None:
        reference = {
            "inplace": core.stream_scale_in_place(1 << qubits, 7)["gb_per_second"],
            "triad": core.stream_triad(1 << qubits, 7)["gb_per_second"],
            "copy": core.stream_copy(1 << qubits, 7)["gb_per_second"],
        }

    achieved = bytes_moved / seconds / 1e9 if seconds > 0 else 0.0
    return {
        **_environment(),
        "experiment": "kernel_bandwidth",
        "kernel": kernel,
        "qubits": qubits,
        "target_qubit": target_qubit,
        "threads": in_force,
        "repeats": repeats,
        "seconds_per_gate": seconds,
        "bytes_per_gate": bytes_moved,
        "gb_per_second": achieved,
        "inplace_gb_per_second": reference["inplace"],
        "triad_gb_per_second": reference["triad"],
        "copy_gb_per_second": reference["copy"],
        # The in-place reference is the comparable one: a gate kernel reads
        # and writes a single array, which is a different traffic shape from
        # copying between two.
        "fraction_of_inplace": achieved / reference["inplace"] if reference["inplace"] else 0.0,
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in KERNEL_RAW_FIELDS})


def run_suite(
    output: Path,
    qubits: int = 24,
    thread_counts: list[int] | None = None,
    kernels: list[str] | None = None,
    repeats: int = 7,
    verbose: bool = True,
) -> Path:
    """Sweep kernels, thread counts and target qubit positions."""
    import os

    from aegisq import native_core

    core = native_core()
    if core is None:
        raise RuntimeError("the native core is not built")

    kernels = kernels or list(_TOUCHED_FRACTION)
    thread_counts = thread_counts or sorted({1, 2, 4, os.cpu_count() or 4})
    rows: list[dict[str, Any]] = []

    for threads in thread_counts:
        core.set_num_threads(threads)
        reference = {
            "inplace": core.stream_scale_in_place(1 << qubits, 7)["gb_per_second"],
            "triad": core.stream_triad(1 << qubits, 7)["gb_per_second"],
            "copy": core.stream_copy(1 << qubits, 7)["gb_per_second"],
        }
        if verbose:
            print(
                f"  {threads} thread(s): in-place {reference['inplace']:.1f} GB/s, "
                f"triad {reference['triad']:.1f}, copy {reference['copy']:.1f}"
            )
        for kernel in kernels:
            # Target qubit 0 gives adjacent pairs; the highest gives pairs half
            # a state vector apart. Both extremes and one in between.
            for target in sorted({1, qubits // 2, qubits - 1}):
                row = measure_kernel(
                    kernel, qubits, target, threads, repeats=repeats, reference=reference
                )
                rows.append(row)
                if verbose:
                    print(
                        f"    {kernel:5s} q{target:<2d} "
                        f"{row['seconds_per_gate'] * 1000:7.2f} ms  "
                        f"{row['gb_per_second']:6.1f} GB/s  "
                        f"{row['fraction_of_inplace'] * 100:5.1f}% of in-place reference"
                    )

    append_rows(output, rows)
    return output
