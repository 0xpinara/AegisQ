"""The placement experiment: default mapping versus communication-aware mapping.

For each circuit family and rank count the same circuit is run twice — once
with the runtime's default placement, once with the placement chosen by the
static mapper — and both runs are recorded as raw measurements. Everything
downstream (tables, plots, the reduction percentages quoted anywhere) is
derived from those rows.

Nothing here computes a "predicted improvement" and presents it as a result.
The prediction is recorded alongside the measurement in the same row so the
two can be compared honestly.
"""

from __future__ import annotations

from pathlib import Path

from aegisq.benchmark.scaling import (
    LaunchResult,
    _echo,
    _runner_args,
    launch,
    resolve_threads,
)

#: Families whose communication patterns differ enough to be worth comparing.
DEFAULT_FAMILIES = ("ghz", "qft", "ising", "grover", "random")


def mapping_experiment(
    families: list[str],
    qubits: int,
    ranks: list[int],
    output: Path,
    *,
    precision: str = "fp64",
    repeats: int = 3,
    shots: int = 0,
    seed: int = 42,
    options: dict[str, list[str]] | None = None,
    thread_policy: str = "fixed-total-cores",
    levers: tuple[str, ...] = ("placement",),
    launches: int = 1,
    verbose: bool = True,
) -> list[LaunchResult]:
    """Run every combination of the requested optimisation levers.

    `levers` selects which dimensions are swept. `("placement",)` reproduces
    the original experiment; `("placement", "fusion")` gives the full 2x2,
    which is what shows whether the two levers are independent or overlap.
    """
    options = options or {}
    results: list[LaunchResult] = []

    mappings = ["default"]
    if "placement" in levers:
        mappings.append("optimized")
    if "windowed" in levers:
        mappings.append("windowed")
    fusions = (False, True) if "fusion" in levers else (False,)

    for family in families:
        for count in ranks:
            # The launch loop sits outside the treatments, so each round
            # visits every treatment once. Two things follow. Any drift in
            # machine state is shared by the whole round instead of landing
            # on whichever treatment happens to run last, and the estimator
            # downstream -- `summarise` takes the minimum wall time over
            # every row for a configuration -- becomes a minimum over
            # independent launches rather than over three repeats inside
            # one. That matters because the interference this machine
            # produces is one-sided: it makes a launch slower, never
            # faster, so the minimum converges on the uncontended runtime.
            # Resampling the A/A calibration puts the gain at roughly
            # 20% -> 10% -> 6% resolution for one, three and five launches.
            for index in range(max(1, launches)):
                for mapping in mappings:
                    for fusion in fusions:
                        args = _runner_args(
                            family,
                            qubits,
                            "mapping_comparison",
                            precision,
                            mapping,
                            repeats,
                            shots,
                            seed,
                            output,
                            options.get(family),
                            thread_policy,
                            fusion,
                            launch_index=index,
                        )
                        result = launch(count, args, threads=resolve_threads(count, thread_policy))
                        if verbose:
                            label = f"{mapping}{'+fusion' if fusion else ''}"
                            suffix = f" launch {index + 1}/{launches}" if launches > 1 else ""
                            _echo(
                                result,
                                f"levers {family} n={qubits} ranks={count} {label}{suffix}",
                            )
                        results.append(result)
    return results
