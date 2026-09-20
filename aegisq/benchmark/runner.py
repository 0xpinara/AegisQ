"""Execute one benchmark configuration and record it as a raw measurement.

Design rules that the rest of the benchmark suite depends on:

1. **Every committed number comes from a run of this module.** Nothing in
   `benchmarks/processed/` or `benchmarks/plots/` may be typed by hand.
2. **Raw rows are append-only.** A raw CSV is a log of measurements, each with
   the environment it was taken in; processed tables are derived from it.
3. **The environment is recorded with the measurement**, not in a README that
   drifts: host, CPU, core count, OS, compiler, MPI library, git commit,
   thread count and precision all live in the row.

The module runs *inside* the MPI world: `mpirun -np 4 python -m
aegisq.benchmark.runner ...` measures a four-rank run and rank 0 writes the
row. Sweeps over rank counts are driven by `aegisq.benchmark.scaling`, which
launches one mpirun per configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegisq import __version__

#: Column order of every raw measurement file.
RAW_FIELDS = [
    # provenance
    "timestamp",
    "hostname",
    "cpu_model",
    "logical_cores",
    "os",
    "python_version",
    "compiler",
    "mpi_library",
    "git_commit",
    "git_dirty",
    "aegisq_version",
    # configuration
    "experiment",
    "circuit_family",
    "circuit_name",
    "qubits",
    "gates",
    "depth",
    "two_qubit_gates",
    "precision",
    "ranks",
    "omp_threads",
    "thread_policy",
    "mapping_strategy",
    "fusion",
    "gates_before_fusion",
    "global_qubits",
    "shots",
    "seed",
    "repeat",
    # measurements
    "wall_seconds",
    "compute_seconds",
    "communication_seconds",
    "bytes_sent",
    "bytes_received",
    "pairwise_exchanges",
    "communicating_gates",
    "local_amplitudes",
    # cost-model predictions, for comparison
    "predicted_bytes",
    "predicted_exchanges",
]


@dataclass
class BenchmarkConfig:
    """One measurable configuration."""

    circuit_family: str
    qubits: int
    experiment: str = "adhoc"
    precision: str = "fp64"
    mapping_strategy: str = "default"
    fusion: bool = False
    window_size: int = 128
    shots: int = 0
    seed: int = 42
    repeats: int = 3
    warmup: int = 1
    thread_policy: str = "unspecified"
    options: dict[str, Any] = field(default_factory=dict)


#: Paths whose contents are *produced by* measuring, and so cannot be part of
#: deciding whether the code that produced a measurement was committed.
#: Without this exclusion every run after the first reports a dirty tree,
#: because writing the first raw CSV dirties it.
_OUTPUT_PATHS = ("benchmarks/", "paper/main.pdf", "paper/tables/")


def git_commit() -> tuple[str, bool]:
    """Current commit, and whether the *code* differs from it.

    Dirtiness here means "the source that produced this measurement is not
    what the commit contains". Generated output does not count: a benchmark
    run necessarily writes files, and a flag that turned itself on partway
    through a sweep would mark every measurement unattributable.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown", False
    if commit.returncode != 0:
        return "unknown", False

    source_changes = [
        line
        for line in status.stdout.splitlines()
        if line.strip() and not any(part in line for part in _OUTPUT_PATHS)
    ]
    return commit.stdout.strip(), bool(source_changes)


#: Provenance columns that every raw file carries, whichever suite wrote it.
#:
#: Each specialised suite used to assemble these itself, and five of the six
#: dropped `git_dirty` -- so most of the repository's raw data could not say
#: whether the tree was modified when it was measured, and the report's
#: "measured from a modified source tree" warning could only ever see the
#: three files that happened to keep the column. The list is checked against
#: every suite's field list by `test_raw_field_lists_carry_provenance`.
PROVENANCE_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "os",
    "aegisq_version",
    "git_commit",
    "git_dirty",
]


def provenance_row() -> dict[str, Any]:
    """Who, where, when and from which source tree -- for any suite.

    Deliberately free of simulator state so the cryptographic and search
    suites can use it unchanged; `environment_row` adds the toolchain
    columns that only apply to a simulation run.
    """
    from aegisq.runtime import hardware

    commit, dirty = git_commit()
    cpu = hardware.cpu_info()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "cpu_model": cpu.extra.get("model", "unknown"),
        "logical_cores": cpu.extra.get("logical_cores", 0),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "aegisq_version": __version__,
        "git_commit": commit,
        "git_dirty": int(dirty),
    }


def environment_row() -> dict[str, Any]:
    """Provenance columns shared by every row of a simulation run."""
    from aegisq import native_core
    from aegisq.runtime.distributed import mpi_library_version

    core = native_core()
    return {
        **provenance_row(),
        "compiler": core.compiler() if core is not None else "none",
        "mpi_library": mpi_library_version(),
    }


def _threads_in_force() -> int:
    """Threads the local kernels will actually use."""
    from aegisq import native_core

    core = native_core()
    return int(core.max_threads()) if core is not None else 1


def build_circuit(config: BenchmarkConfig):
    from aegisq.algorithms import build_circuit as make

    return make(config.circuit_family, config.qubits, **config.options)


def measure(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Run one configuration `repeats` times and return the raw rows.

    A warm-up run is executed first and discarded: the first touch of a fresh
    shard pays page-fault and allocation costs that say nothing about the
    steady-state cost of the circuit.
    """
    from aegisq.compiler import CommunicationCostModel, optimize_placement
    from aegisq.compiler.cost_model import default_global_qubits
    from aegisq.runtime import Simulator
    from aegisq.runtime.distributed import is_distributed, preferred_backend, rank, world_size

    circuit = build_circuit(config)
    gates_before_fusion = len(circuit)
    if config.fusion:
        from aegisq.compiler import fuse

        circuit = fuse(circuit)

    ranks = world_size()
    backend = preferred_backend()

    mapping = None
    windowed_plan = None
    if config.mapping_strategy == "optimized" and is_distributed():
        placement = optimize_placement(circuit, ranks, config.precision)
        mapping = list(placement.mapping)
        global_qubits = list(placement.global_qubits)
    elif config.mapping_strategy == "windowed" and is_distributed():
        from aegisq.compiler.dynamic_mapper import apply_plan, plan_dynamic_placement

        planning_model = CommunicationCostModel(circuit.num_qubits, ranks, config.precision)
        windowed_plan = plan_dynamic_placement(
            circuit, planning_model, window_size=config.window_size
        )
        # Slots are baked into the rewritten operands, so the runtime keeps
        # the identity mapping and the plan's own SWAPs do the moving.
        circuit = apply_plan(circuit, windowed_plan, planning_model)
        global_qubits = list(windowed_plan.static_assignment)
    else:
        global_qubits = list(default_global_qubits(circuit.num_qubits, ranks))

    model = CommunicationCostModel(circuit.num_qubits, ranks, config.precision)
    if windowed_plan is not None:
        # The rewritten circuit runs under the default placement; predicting
        # it that way is what the plan itself claims.
        prediction = model.estimate(circuit, default_global_qubits(circuit.num_qubits, ranks))
    else:
        prediction = model.estimate(circuit, global_qubits)

    options: dict[str, Any] = {}
    if mapping is not None:
        options["mapping"] = mapping

    simulator = Simulator(backend, precision=config.precision, **options)
    environment = environment_row()
    rows: list[dict[str, Any]] = []

    for repeat in range(-config.warmup, config.repeats):
        started = time.perf_counter()
        result = simulator.run(
            circuit,
            shots=config.shots,
            seed=config.seed,
            save_statevector=False,
        )
        wall = time.perf_counter() - started
        if repeat < 0:
            continue  # warm-up
        metrics = result.metrics

        rows.append(
            {
                **environment,
                "experiment": config.experiment,
                "circuit_family": config.circuit_family,
                "circuit_name": circuit.name,
                "qubits": circuit.num_qubits,
                "gates": len(circuit),
                "depth": circuit.depth(),
                "two_qubit_gates": circuit.two_qubit_gate_count(),
                "precision": config.precision,
                "ranks": ranks,
                "omp_threads": _threads_in_force(),
                "thread_policy": config.thread_policy,
                "mapping_strategy": config.mapping_strategy,
                "fusion": "on" if config.fusion else "off",
                "gates_before_fusion": gates_before_fusion,
                "global_qubits": " ".join(str(q) for q in global_qubits),
                "shots": config.shots,
                "seed": config.seed,
                "repeat": repeat,
                "wall_seconds": wall,
                "compute_seconds": metrics.get("compute_seconds", 0.0),
                "communication_seconds": metrics.get("communication_seconds", 0.0),
                "bytes_sent": metrics.get("bytes_sent", 0),
                "bytes_received": metrics.get("bytes_received", 0),
                "pairwise_exchanges": metrics.get("pairwise_exchanges", 0),
                "communicating_gates": metrics.get("communicating_gates", 0),
                "local_amplitudes": metrics.get("local_amplitudes", 2**circuit.num_qubits),
                "predicted_bytes": prediction.bytes_sent,
                "predicted_exchanges": prediction.pairwise_exchanges,
            }
        )

    return rows if rank() == 0 else []


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append measurements to a raw CSV, writing the header once."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RAW_FIELDS})


def default_raw_path(experiment: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "raw"
    stamp = time.strftime("%Y%m%d")
    host = socket.gethostname().split(".")[0]
    return root / f"{experiment}_{stamp}_{host}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aegisq.benchmark.runner",
        description=(
            "Measure one benchmark configuration in the current MPI world. "
            "Launch under mpirun to measure a distributed run."
        ),
    )
    parser.add_argument("--circuit", required=True, help="benchmark family name")
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--experiment", default="adhoc")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument(
        "--mapping", choices=("default", "optimized", "windowed"), default="default"
    )
    parser.add_argument(
        "--window-size", type=int, default=128, help="gates per window for --mapping windowed"
    )
    parser.add_argument(
        "--fuse", action="store_true", help="fuse single-qubit runs before executing"
    )
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="OpenMP threads per rank (0 keeps the runtime default)",
    )
    parser.add_argument(
        "--thread-policy",
        default="unspecified",
        help=(
            "label describing how --threads was chosen: one-thread-per-rank "
            "(cores grow with ranks) or fixed-total-cores (ranks x threads held "
            "at the core count)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--option", action="append", metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, help="raw CSV to append to")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    options: dict[str, Any] = {}
    for item in args.option or []:
        key, _, value = item.partition("=")
        try:
            options[key] = int(value)
        except ValueError:
            options[key] = value

    if args.threads > 0:
        from aegisq import native_core

        core = native_core()
        if core is not None:
            in_force = core.set_num_threads(args.threads)
            os.environ["OMP_NUM_THREADS"] = str(in_force)

    config = BenchmarkConfig(
        circuit_family=args.circuit,
        qubits=args.qubits,
        experiment=args.experiment,
        precision=args.precision,
        mapping_strategy=args.mapping,
        fusion=args.fuse,
        window_size=args.window_size,
        shots=args.shots,
        seed=args.seed,
        repeats=args.repeats,
        warmup=args.warmup,
        thread_policy=args.thread_policy,
        options=options,
    )

    rows = measure(config)
    if not rows:
        return 0  # non-zero ranks stay silent

    output = args.output or default_raw_path(args.experiment)
    append_rows(output, rows)
    if args.print_json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        best = min(row["wall_seconds"] for row in rows)
        print(
            f"{config.circuit_family} n={rows[0]['qubits']} ranks={rows[0]['ranks']} "
            f"{config.mapping_strategy}"
            f"{'+fusion' if config.fusion else ''}: best {best * 1000:.1f} ms, "
            f"{rows[0]['bytes_sent'] / 2**20:.2f} MiB sent -> {output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
