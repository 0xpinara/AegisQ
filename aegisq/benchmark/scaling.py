"""Scaling sweeps: one `mpirun` per configuration, one raw row per repeat.

A sweep cannot run inside a single MPI world, because the world size is fixed
when `mpirun` starts. The driver therefore spawns one launcher invocation per
rank count and appends every result to the same raw CSV.

Thread accounting
-----------------
On a single node, `ranks x threads` is held equal to the physical core count
by default, so that a 1-rank run and an 8-rank run are given the same hardware
rather than the 8-rank run quietly getting eight times as much of it. The
thread count is passed to the runner as an argument (not an environment
variable) because MPI launchers differ in how they propagate the environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaunchResult:
    ranks: int
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def mpi_launcher() -> str | None:
    for candidate in ("mpirun", "mpiexec", "srun"):
        if shutil.which(candidate):
            return candidate
    return None


def supports_oversubscribe(launcher: str) -> bool:
    """Laptops have fewer cores than the rank counts a sweep wants to try."""
    try:
        help_text = subprocess.run(
            [launcher, "--help"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--oversubscribe" in (help_text.stdout + help_text.stderr)


def threads_for(ranks: int, cores: int | None = None) -> int:
    """Threads per rank so that ranks x threads stays near the core count."""
    cores = cores or os.cpu_count() or 1
    return max(1, cores // max(1, ranks))


#: Thread policies a sweep can use. They answer different questions and must
#: never be mixed inside one table.
#:
#: ``one-thread-per-rank``
#:     Each rank gets a single thread, so total compute grows with the rank
#:     count. This is the classic strong/weak-scaling setup.
#: ``fixed-total-cores``
#:     ``ranks x threads`` is held at the core count, so every configuration
#:     is given the *same* hardware. This measures the cost of partitioning
#:     rather than the benefit of adding processors.
THREAD_POLICIES = ("one-thread-per-rank", "fixed-total-cores")


def resolve_threads(ranks: int, policy: str, cores: int | None = None) -> int:
    if policy == "one-thread-per-rank":
        return 1
    if policy == "fixed-total-cores":
        return threads_for(ranks, cores)
    raise ValueError(f"unknown thread policy {policy!r}; use one of {THREAD_POLICIES}")


def launch(
    ranks: int,
    runner_args: list[str],
    python: str | None = None,
    threads: int | None = None,
    timeout: float = 3600.0,
) -> LaunchResult:
    """Run `aegisq.benchmark.runner` in a world of `ranks` processes."""
    python = python or sys.executable
    threads = threads if threads is not None else threads_for(ranks)
    command: list[str] = []

    if ranks > 1:
        launcher = mpi_launcher()
        if launcher is None:
            raise RuntimeError("no MPI launcher found; multi-rank sweeps need mpirun/mpiexec")
        command += [launcher]
        if launcher != "srun" and supports_oversubscribe(launcher):
            command += ["--oversubscribe"]
        command += ["-np", str(ranks)]

    command += [python, "-m", "aegisq.benchmark.runner", *runner_args, "--threads", str(threads)]

    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    return LaunchResult(ranks, completed.returncode, completed.stdout, completed.stderr)


def _runner_args(
    circuit: str,
    qubits: int,
    experiment: str,
    precision: str,
    mapping: str,
    repeats: int,
    shots: int,
    seed: int,
    output: Path,
    options: list[str] | None,
    thread_policy: str = "unspecified",
    fusion: bool = False,
) -> list[str]:
    args = [
        "--circuit",
        circuit,
        "--qubits",
        str(qubits),
        "--experiment",
        experiment,
        "--precision",
        precision,
        "--mapping",
        mapping,
        "--repeats",
        str(repeats),
        "--shots",
        str(shots),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--thread-policy",
        thread_policy,
    ]
    if fusion:
        args.append("--fuse")
    for option in options or []:
        args += ["--option", option]
    return args


def strong_scaling(
    circuit: str,
    qubits: int,
    ranks: list[int],
    output: Path,
    *,
    precision: str = "fp64",
    mapping: str = "default",
    repeats: int = 3,
    shots: int = 0,
    seed: int = 42,
    options: list[str] | None = None,
    thread_policy: str = "one-thread-per-rank",
    verbose: bool = True,
) -> list[LaunchResult]:
    """Fixed problem size, increasing rank count."""
    results = []
    for count in ranks:
        args = _runner_args(
            circuit,
            qubits,
            "strong_scaling",
            precision,
            mapping,
            repeats,
            shots,
            seed,
            output,
            options,
            thread_policy,
        )
        result = launch(count, args, threads=resolve_threads(count, thread_policy))
        if verbose:
            _echo(result, f"strong {circuit} n={qubits} ranks={count}")
        results.append(result)
    return results


def weak_scaling(
    circuit: str,
    base_qubits: int,
    ranks: list[int],
    output: Path,
    *,
    precision: str = "fp64",
    mapping: str = "default",
    repeats: int = 3,
    shots: int = 0,
    seed: int = 42,
    options: list[str] | None = None,
    thread_policy: str = "one-thread-per-rank",
    verbose: bool = True,
) -> list[LaunchResult]:
    """Grow the problem with the rank count so each shard stays the same size.

    Doubling the ranks adds one qubit, which doubles the state and leaves
    `2^(n-p)` amplitudes per rank unchanged.
    """
    results = []
    for count in ranks:
        qubits = base_qubits + (count.bit_length() - 1)
        args = _runner_args(
            circuit,
            qubits,
            "weak_scaling",
            precision,
            mapping,
            repeats,
            shots,
            seed,
            output,
            options,
            thread_policy,
        )
        result = launch(count, args, threads=resolve_threads(count, thread_policy))
        if verbose:
            _echo(result, f"weak {circuit} n={qubits} ranks={count}")
        results.append(result)
    return results


def _echo(result: LaunchResult, label: str) -> None:
    if result.ok:
        print(
            f"  [ok] {label}: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'done'}"
        )
    else:
        print(f"  [FAILED] {label} (exit {result.returncode})")
        tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        for line in tail:
            print(f"        {line}")
