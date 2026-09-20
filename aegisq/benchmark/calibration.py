"""How small a wall-time difference can this harness actually see?

Every wall-time claim in this repository is a difference between two
measurements taken in two separate process launches. The question that
governs all of them is not answered by running more repeats inside one
launch -- that measures how steady a single process is, which turns out
to be the easy part. It is answered by launching the *same* configuration
twice and looking at how different the two come out.

That is an A/A test, and here it is decisive. The placement table once
reported a 30.8% wall-time change for a GHZ circuit whose byte count the
optimiser had left exactly unchanged. Running GHZ against itself at 8
ranks produces apparent changes of -35% and +51%. The reported effect
was not merely unsupported; it was the same size as the instrument's own
scatter.

So the null distribution is measured rather than assumed. For each
configuration the same run is launched twice, `trials` times over, and
the apparent change

    (second - first) / first

is recorded. Nothing about that quantity should be non-zero. Its spread
is the resolution of the harness for that configuration, and a measured
effect has to clear it before it means anything.

The criterion built on this is non-parametric and needs no assumption
about the shape of the distribution: an observed change that exceeds
every one of `n` null trials has probability at most `1 / (n + 1)` of
arising by chance if the two conditions were exchangeable. With ten
trials that is `p <= 0.09`; it is a weak claim, honestly labelled, and
much stronger than the nothing that preceded it.

Relative resolution improves with how long the circuit runs -- a GHZ
chain finishing in 8 ms is far harder to time than a Grover circuit
taking 450 ms -- so the null is measured per circuit family and rank
count rather than once for the machine.
"""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path
from typing import Any

CALIBRATION_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "logical_cores",
    "os",
    "python_version",
    "aegisq_version",
    "git_commit",
    "git_dirty",
    "experiment",
    "circuit_family",
    "qubits",
    "precision",
    "ranks",
    "thread_policy",
    "repeats_per_launch",
    "trial",
    "first_wall_s",
    "second_wall_s",
    "apparent_change",
]

#: Same families the placement experiment uses, so every cell of that
#: table has a null measured under matching conditions.
DEFAULT_FAMILIES = ("ghz", "qft", "ising", "grover", "random")


def _environment() -> dict[str, Any]:
    from aegisq.benchmark.runner import provenance_row

    return provenance_row()


def _launch_once(
    family: str,
    qubits: int,
    ranks: int,
    precision: str,
    repeats: int,
    thread_policy: str,
    options: list[str] | None,
    directory: Path,
    tag: str,
) -> float:
    """Run one launch and return its best wall time.

    Best-of-repeats within the launch, matching exactly how the placement
    experiment reduces a launch to a number. The point is to calibrate
    that estimator, so it has to be the same estimator.
    """
    from aegisq.benchmark.scaling import _runner_args, launch, resolve_threads

    output = directory / f"{tag}.csv"
    args = _runner_args(
        family,
        qubits,
        "calibration",
        precision,
        "default",
        repeats,
        0,
        42,
        output,
        options,
        thread_policy,
        False,
    )
    result = launch(ranks, args, threads=resolve_threads(ranks, thread_policy))
    if not result.ok:
        raise RuntimeError(f"calibration launch failed at {ranks} ranks:\n{result.stderr}")

    import pandas as pd

    frame = pd.read_csv(output)
    output.unlink(missing_ok=True)
    return float(frame["wall_seconds"].min())


def measure_null(
    family: str,
    qubits: int,
    ranks: int,
    trials: int = 10,
    precision: str = "fp64",
    repeats: int = 3,
    thread_policy: str = "fixed-total-cores",
    options: list[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Launch one configuration against itself, `trials` times."""
    rows: list[dict[str, Any]] = []
    environment = _environment()

    with tempfile.TemporaryDirectory(prefix="aegisq-calibration-") as scratch:
        directory = Path(scratch)
        for trial in range(trials):
            first = _launch_once(
                family,
                qubits,
                ranks,
                precision,
                repeats,
                thread_policy,
                options,
                directory,
                f"{family}-{ranks}-{trial}-a",
            )
            second = _launch_once(
                family,
                qubits,
                ranks,
                precision,
                repeats,
                thread_policy,
                options,
                directory,
                f"{family}-{ranks}-{trial}-b",
            )
            rows.append(
                {
                    **environment,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "experiment": "calibration",
                    "circuit_family": family,
                    "qubits": qubits,
                    "precision": precision,
                    "ranks": ranks,
                    "thread_policy": thread_policy,
                    "repeats_per_launch": repeats,
                    "trial": trial,
                    "first_wall_s": first,
                    "second_wall_s": second,
                    "apparent_change": (second - first) / first if first else 0.0,
                }
            )

    if verbose and rows:
        changes = [abs(row["apparent_change"]) for row in rows]
        print(
            f"  {family:7s} {ranks} ranks: {len(rows)} null trials, "
            f"median |change| {sorted(changes)[len(changes) // 2] * 100:5.1f}%, "
            f"worst {max(changes) * 100:5.1f}%"
        )
    return rows


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALIBRATION_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CALIBRATION_RAW_FIELDS})


def run_suite(
    output: Path,
    qubits: int = 20,
    ranks: list[int] | None = None,
    families: list[str] | None = None,
    trials: int = 10,
    precision: str = "fp64",
    repeats: int = 3,
    thread_policy: str = "fixed-total-cores",
    options: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> Path:
    """Measure the null distribution for every cell of the placement table."""
    ranks = ranks or [2, 4, 8]
    families = list(families or DEFAULT_FAMILIES)
    options = options or {}
    rows: list[dict[str, Any]] = []

    for count in ranks:
        for family in families:
            rows.extend(
                measure_null(
                    family,
                    qubits,
                    count,
                    trials=trials,
                    precision=precision,
                    repeats=repeats,
                    thread_policy=thread_policy,
                    options=options.get(family),
                    verbose=verbose,
                )
            )

    append_rows(output, rows)
    return output
