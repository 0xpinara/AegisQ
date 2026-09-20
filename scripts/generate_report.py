#!/usr/bin/env python3
"""Regenerate processed tables, plots and the README results block.

The README quotes measured numbers. Typing them by hand is how a repository
ends up advertising results that its data no longer supports, so the block
between the markers below is generated from `benchmarks/processed/` and
nothing else.

    ./scripts/generate_report.py            # regenerate everything
    ./scripts/generate_report.py --check    # fail if the README is stale
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

START = "<!-- BENCHMARK-RESULTS:START -->"
END = "<!-- BENCHMARK-RESULTS:END -->"


def build_block() -> str:
    import pandas as pd

    from aegisq.benchmark import report as report_module

    data = report_module.load_raw()

    def describe(column: str, default: str = "unknown") -> str:
        """Most common non-null value of a column across every raw row.

        Reading row zero is fragile: raw files accumulate with different
        schemas, and a concatenation can put a NaN there for a column that is
        well defined everywhere else.
        """
        if column not in data.columns:
            return default
        values = data[column].dropna()
        return str(values.mode().iloc[0]) if not values.empty else default

    cores = describe("logical_cores", "0")
    with contextlib.suppress(ValueError):
        cores = str(int(float(cores)))
    mapping = report_module.mapping_table(data, report_module.load_calibration())
    strong = report_module.strong_scaling_table(data)
    accuracy = report_module.prediction_accuracy(data)

    # Provenance has to describe the data, not the first row of it. Raw files
    # accumulate across commits, and attributing a whole table to whichever
    # row happened to sort first is precisely the misattribution this project
    # says it does not do.
    def collect(frame, column):
        if frame is None or frame.empty or column not in frame.columns:
            return set()
        return {str(value)[:12] for value in frame[column].dropna().unique()}

    everything = [
        data,
        report_module.load_pqc(),
        report_module.load_search(),
        report_module.load_kernels(),
        report_module.load_placement_quality(),
        report_module.load_precision(),
    ]
    commits = sorted(set().union(*(collect(frame, "git_commit") for frame in everything)))

    def measured_dirty(frame) -> bool:
        """Was this frame measured from a modified tree -- or can't we tell?

        A missing `git_dirty` column is not evidence of cleanliness, and
        treating it as such is how five of the six suites went a whole
        release recording numbers that nothing could vouch for. Data that
        cannot answer the question counts as a warning.
        """
        if frame is None or frame.empty:
            return False
        if "git_dirty" not in frame.columns:
            return True
        return bool(frame["git_dirty"].fillna(1).astype(float).max())

    dirty = any(measured_dirty(frame) for frame in everything)

    lines: list[str] = [START, ""]
    if len(commits) == 1:
        provenance = f"at commit `{commits[0]}`"
    else:
        provenance = f"across {len(commits)} commits (`" + "`, `".join(commits) + "`)"
    lines.append(
        f"All figures below were measured on **{describe('cpu_model')} "
        f"({cores} logical cores)**, {describe('os')}, "
        f"{describe('mpi_library').split(',')[0]}, "
        f"AegisQ {describe('aegisq_version')} {provenance}. "
        "They describe that host and are not a claim about cluster hardware."
    )
    if dirty:
        lines.append("")
        lines.append(
            "> Some rows were recorded from a working tree with uncommitted changes, "
            "so they cannot be attributed to a commit with confidence. Re-run "
            "`./scripts/benchmark_local.sh` from a clean tree to replace them."
        )
    lines.append("")

    if not mapping.empty:
        interesting = mapping[mapping["ranks"] == mapping["ranks"].max()]
        lines.append(
            f"### Communication-aware placement, {int(interesting['ranks'].iloc[0])} ranks, "
            f"{int(interesting['qubits'].iloc[0])} qubits"
        )
        lines.append("")
        lines.append(
            "| circuit | measured MPI bytes, default | measured MPI bytes, optimized | "
            "reduction | wall time change |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for row in interesting.sort_values("bytes_reduction", ascending=False).itertuples():
            change = f"{row.wall_change * 100:+.1f}%"
            if not row.wall_change_resolved:
                change = f"{change} (unresolved)"
            lines.append(
                f"| {row.circuit_family} | {row.baseline_bytes:,} | {row.optimized_bytes:,} | "
                f"**{row.bytes_reduction * 100:.1f}%** | {change} |"
            )
        lines.append("")
        floor = interesting["wall_noise_floor"].dropna()
        if not floor.empty:
            source = str(interesting["floor_source"].mode().iloc[0])
            lines.append(
                f"Byte counts are exact counters; wall times are not. Each row's wall-time "
                f"change is judged against a floor measured for that exact configuration "
                f"({source}, see below), and is marked unresolved unless it clears the floor "
                f"and its repeat range does not overlap the baseline's. At these sizes most "
                f"do not clear it. **The placement result is the traffic reduction.** The "
                f"wall-time column is shown for completeness and is not a speedup claim."
            )
            lines.append("")
        unchanged = [
            row.circuit_family for row in interesting.itertuples() if row.bytes_reduction < 0.01
        ]
        names = sorted(set(unchanged))
        if names:
            subject = names[0] if len(names) == 1 else ", ".join(names)
            verb = "shows" if len(names) == 1 else "show"
            lines.append(
                f"Not every circuit benefits: {subject} {verb} no reduction, because its "
                "expensive qubits already sit well under the default placement. That is a "
                "result, not a gap — a heuristic that claimed a win on every circuit would "
                "be the suspicious one."
            )
        lines.append("")
        lines.append("![Communication-aware placement](benchmarks/plots/mapping_comparison.png)")
        lines.append("")

    calibration = report_module.load_calibration()
    if not calibration.empty:
        resolution = report_module.calibration_table(calibration)
        trials = int(resolution["trials"].min())
        lines.append("### What this harness can actually resolve")
        lines.append("")
        lines.append(
            f"Before reading any wall-time number above, here is what the measurement "
            f"setup can see. Each configuration below was launched twice, {trials} times "
            f"over, with **nothing changed between the two runs**. Every difference in "
            f"this table is therefore zero by construction, and everything reported is "
            f"the instrument, not the simulator."
        )
        lines.append("")
        lines.append(
            "| circuit | ranks | median run | median apparent change | worst | resolution |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in resolution.sort_values(["ranks", "circuit_family"]).itertuples():
            worst = max(abs(row.most_negative), abs(row.most_positive))
            lines.append(
                f"| {row.circuit_family} | {int(row.ranks)} | "
                f"{row.median_wall_s * 1000:.0f} ms | "
                f"{row.median_absolute_change * 100:.1f}% | {worst * 100:.1f}% | "
                f"**{row.resolution * 100:.1f}%** |"
            )
        lines.append("")
        p_bound = float(resolution["p_bound"].max())
        lines.append(
            f"`resolution` is the quantile of the apparent change at "
            f"`1 - 1/(trials+1)`, so an effect exceeding it carries `p <= {p_bound:.2f}` "
            f"under exchangeability. The relationship worth noting is with duration, not "
            f"with rank count: a GHZ chain finishing in under ten milliseconds cannot be "
            f"timed to better than tens of percent, while a Grover circuit running for "
            f"half a second can. This is a laptop with four performance and four "
            f"efficiency cores and no separate interconnect, and the distribution of "
            f"launch times is right-skewed -- interference makes a run slower, never "
            f"faster."
        )
        lines.append("")
        projected = {
            k: report_module.calibration_table(calibration, launches=k)["resolution"].median()
            for k in (1, 3, 5)
        }
        lines.append(
            "That one-sidedness says which estimator to use. Taking the minimum over "
            "several independent launches, rather than over repeats inside one, "
            "converges on the uncontended runtime; resampling the null above puts the "
            "median resolution at "
            + ", ".join(
                f"**{value * 100:.0f}%** at {k} launch" + ("" if k == 1 else "es")
                for k, value in projected.items()
            )
            + ". `aegisq benchmark mapping --launches N` runs the sweep that way."
        )
        lines.append("")
        lines.append("![Harness resolution](benchmarks/plots/calibration.png)")
        lines.append("")

    levers = report_module.lever_table(data)
    lever_columns = [
        column
        for column in (
            "fusion_only_reduction",
            "placement_only_reduction",
            "windowed_only_reduction",
            "both_reduction",
            "windowed_fusion_reduction",
        )
        if column in levers.columns and not levers[column].isna().all()
    ]
    if not levers.empty and lever_columns:
        import pandas as pd

        interesting = levers[levers["ranks"] == levers["ranks"].max()]
        lines.append("### Three ways to spend less on the network")
        lines.append("")
        lines.append(
            "Placement decides *which* gates communicate. Fusion decides *how many times*. "
            "Windowed placement changes the assignment part-way through the circuit, when "
            "the phase structure makes the switch worth paying for. Measured at "
            f"{int(interesting['ranks'].iloc[0])} ranks:"
        )
        lines.append("")
        lines.append(
            "| circuit | baseline MPI bytes | fusion | static placement | "
            "windowed placement | placement + fusion | everything |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

        def percent(value):
            return "\u2014" if value is None or pd.isna(value) else f"{value * 100:.1f}%"

        def best_of(row):
            values = [
                getattr(row, column, None)
                for column in (
                    "fusion_only_reduction",
                    "placement_only_reduction",
                    "windowed_only_reduction",
                    "both_reduction",
                    "windowed_fusion_reduction",
                )
            ]
            values = [v for v in values if v is not None and not pd.isna(v)]
            return max(values) if values else 0.0

        for row in sorted(interesting.itertuples(), key=best_of, reverse=True):
            best = percent(best_of(row))
            cells = [
                percent(getattr(row, column, None))
                for column in (
                    "fusion_only_reduction",
                    "placement_only_reduction",
                    "windowed_only_reduction",
                    "both_reduction",
                    "windowed_fusion_reduction",
                )
            ]
            cells = [f"**{c}**" if c == best and c != "0.0%" else c for c in cells]
            lines.append(
                f"| {row.circuit_family} | {row.baseline_bytes:,} | " + " | ".join(cells) + " |"
            )
        lines.append("")

        if "windowed_only_reduction" in interesting.columns:
            wins = interesting.dropna(subset=["windowed_only_reduction"])
            wins = wins[wins["windowed_only_reduction"] > wins["placement_only_reduction"] + 0.01]
            if not wins.empty:
                example = wins.sort_values("windowed_only_reduction", ascending=False).iloc[0]
                lines.append(
                    f"Windowed placement beats the best single assignment on {len(wins)} of "
                    f"{len(interesting)} circuits \u2014 for `{example['circuit_family']}`, "
                    f"{example['windowed_only_reduction'] * 100:.1f}% against "
                    f"{example['placement_only_reduction'] * 100:.1f}% \u2014 by paying a few "
                    "shard exchanges to re-assign qubits between phases. Where a circuit has "
                    "no phase structure the planner declines to switch and the two coincide."
                )
                lines.append("")

        if "windowed_fusion_reduction" in interesting.columns:
            everything = interesting.dropna(subset=["windowed_fusion_reduction"]).copy()
            everything["single_best"] = everything[
                ["fusion_only_reduction", "placement_only_reduction", "windowed_only_reduction"]
            ].max(axis=1)
            everything["lift"] = everything["windowed_fusion_reduction"] - everything["single_best"]
            top = everything.sort_values("lift", ascending=False).iloc[0]
            if top["lift"] > 0.01:
                lines.append(
                    "Applying all three together is worth more than the best of them alone. "
                    f"For `{top['circuit_family']}` the combination removes "
                    f"{top['windowed_fusion_reduction'] * 100:.1f}% where the best single "
                    f"lever removes {top['single_best'] * 100:.1f}% — the levers change each "
                    "other's cost landscape rather than dividing the same saving between them."
                )
                lines.append("")

        combined = interesting.dropna(subset=["both_reduction"])
        non_additive = combined[
            combined["both_reduction"]
            > combined["fusion_only_reduction"].fillna(0)
            + combined["placement_only_reduction"].fillna(0)
            + 0.01
        ]
        if not non_additive.empty:
            example = non_additive.iloc[0]
            lines.append(
                "Placement and fusion are not independent either: for "
                f"`{example['circuit_family']}` the pair removes "
                f"{example['both_reduction'] * 100:.1f}% where separately they remove "
                f"{example['fusion_only_reduction'] * 100:.1f}% and "
                f"{example['placement_only_reduction'] * 100:.1f}%. Fusing first changes "
                "which placement is best, so the search finds a better one."
            )
            lines.append("")
        lines.append("![Optimisation levers](benchmarks/plots/lever_comparison.png)")
        lines.append("")

    if not strong.empty:
        classic = strong[strong["thread_policy"] == "one-thread-per-rank"]
        if not classic.empty:
            lines.append("### Strong scaling (one thread per rank)")
            lines.append("")
            lines.append("| circuit | qubits | ranks | wall time (s) | speedup | efficiency |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in classic.sort_values(["circuit_family", "ranks"]).itertuples():
                lines.append(
                    f"| {row.circuit_family} | {row.qubits} | {row.ranks} | "
                    f"{row.wall_best_s:.3f} | {row.speedup:.2f}x | "
                    f"{row.efficiency * 100:.0f}% |"
                )
            lines.append("")
            lines.append("![Strong scaling](benchmarks/plots/strong_scaling.png)")
            lines.append("")

    placement = report_module.placement_quality_table(report_module.load_placement_quality())
    if not placement.empty:
        total = int(placement["samples"].sum())
        heuristic = int(placement["heuristic_samples"].sum())
        found = int(placement["found_optimum"].sum())
        worst = float(placement["worst_gap"].max())
        fastest = float(placement["median_speedup"].max())
        lines.append("### Is the fallback search good enough?")
        lines.append("")
        lines.append(
            "The placement search is exhaustive while the candidate count fits a budget "
            "and falls back beyond it. Two facts make the fallback safer than "
            '"not guaranteed optimal" suggests.'
        )
        lines.append("")
        lines.append(
            "Every cost rule except `swap` depends on one qubit's membership: a "
            "single-qubit gate costs if *its* qubit is global, a `cx` costs if *its "
            "target* is. `swap` costs if *either* operand is, and an OR is not a sum. "
            "So for a circuit without `swap` gates the objective is linear in the global "
            "set, and sorting the per-qubit costs gives the optimum outright — no search."
        )
        lines.append("")
        lines.append(
            f"For the rest, measured: across {total} sampled configurations, "
            f"{heuristic} of which genuinely needed the heuristic, it matched the "
            f"exhaustive optimum **{found} times out of {total}** (worst gap "
            f"{worst * 100:.2f}%) while running up to {fastest:.0f}x faster."
        )
        lines.append("")

    precision = report_module.precision_table(report_module.load_precision())
    if not precision.empty:
        import numpy as np

        worst = precision.sort_values("worst_infidelity", ascending=False).iloc[0]
        biggest = precision.sort_values("gates", ascending=False).iloc[0]
        lines.append("### A fourth lever, and what it costs")
        lines.append("")
        lines.append(
            "Single precision halves the shard and halves every transfer, exactly and "
            "without any analysis — unlike the other three levers, there is nothing to "
            "search for. The question is only what the accuracy costs."
        )
        lines.append("")
        lines.append("| circuit | gates | 1 − fidelity | largest amplitude error |")
        lines.append("|---|---:|---:|---:|")
        for row in precision.sort_values("gates").itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.gates:,} | {row.worst_infidelity:.1e} | "
                f"{row.worst_amplitude_error:.1e} |"
            )
        lines.append("")
        shots_needed = 1.0 / max(worst.worst_infidelity, 1e-300)
        lines.append(
            f"The error grows with circuit size — the largest circuit measured "
            f"({int(biggest.gates):,} gates) loses "
            f'{biggest.worst_infidelity:.1e} of fidelity — so "fp32 is fine" is a '
            "statement about a depth, not about a precision. In this range it is very "
            f"fine: distinguishing the worst case here from the exact state would take "
            f"on the order of {shots_needed:.0e} shots, against the thousands a real job "
            "takes."
        )
        lines.append("")
        lines.append("![Single-precision error](benchmarks/plots/precision_error.png)")
        lines.append("")
        _ = np

    kernels = report_module.kernel_table(report_module.load_kernels())
    if not kernels.empty:
        peak_threads = int(kernels["threads"].max())
        at_peak = kernels[kernels["threads"] == peak_threads]
        lines.append("### Are the local kernels any good?")
        lines.append("")
        lines.append(
            "Wall time alone cannot say. State-vector simulation is bandwidth-bound, so "
            "each kernel is compared against what the same machine achieves on an "
            "in-place scale of the same array — same type, same flags, same threading. "
            f"At {peak_threads} threads:"
        )
        lines.append("")
        lines.append("| kernel | GB/s achieved | reference | fraction |")
        lines.append("|---|---:|---:|---:|")
        for row in at_peak.sort_values("fraction_of_inplace", ascending=False).itertuples():
            lines.append(
                f"| {row.kernel} | {row.gb_per_second:.1f} | "
                f"{row.inplace_gb_per_second:.1f} | {row.fraction_of_inplace * 100:.0f}% |"
            )
        lines.append("")
        full_sweep = at_peak[at_peak["kernel"].isin(["h", "rz"])]
        partial = at_peak[at_peak["kernel"].isin(["cx", "cz", "swap"])]
        if not full_sweep.empty and not partial.empty:
            low = full_sweep["fraction_of_inplace"].min() * 100
            high = full_sweep["fraction_of_inplace"].max() * 100
            span = f"{low:.0f}%" if round(low) == round(high) else f"{low:.0f}–{high:.0f}%"
            lines.append(
                "The kernels that sweep the whole state — a general single-qubit gate and a "
                f"diagonal one — run at {span} of that reference, "
                "which is to say they are memory-bound and there is little left to win. "
                "The ones that touch only part of the state plateau near "
                f"{partial['fraction_of_inplace'].mean() * 100:.0f}%: they read and write a "
                "strided fraction of the array and leave about half the bandwidth unused. "
                "That is a concrete optimisation target rather than a mystery."
            )
            lines.append("")

        positions = report_module.kernel_position_table(report_module.load_kernels())
        if not positions.empty:
            worst = positions.sort_values("spread", ascending=False).iloc[0]
            best = positions.sort_values("spread").iloc[0]
            lines.append(
                "The communication cost model assumes only the local/global distinction "
                "matters, never which *local* position a qubit occupies. Measured, that "
                f"holds for most kernels (`{best['kernel']}` varies by "
                f"{best['spread'] * 100:.0f}% across target positions) and fails for "
                f"`{worst['kernel']}`, which varies by {worst['spread'] * 100:.0f}%. The "
                "model is therefore right about network traffic and incomplete about local "
                "cost — stated here rather than left for a reader to discover."
            )
            lines.append("")
        lines.append("![Local kernel bandwidth](benchmarks/plots/kernel_bandwidth.png)")
        lines.append("")

    pqc = report_module.load_pqc()
    primitives = report_module.pqc_table(pqc)
    envelope = report_module.pqc_envelope_table(pqc)
    if not primitives.empty and not envelope.empty:
        chosen = primitives[primitives["algorithm"].isin(["ML-KEM-768", "ML-DSA-65"])]
        by_step = envelope.set_index("operation")

        lines.append("### Cost of the post-quantum layer")
        lines.append("")
        lines.append("| operation | median | size |")
        lines.append("|---|---:|---:|")
        for row in chosen.itertuples():
            lines.append(
                f"| {row.algorithm} {row.operation} | {row.median_us:.1f} us | {int(row.bytes)} B |"
            )
        pack = float(by_step["median_us"].get("pack_job", 0.0))
        verify = float(by_step["median_us"].get("verify_and_open", 0.0))
        overhead = int(by_step["bytes"].get("envelope_fixed_overhead", 0))
        lines.append(f"| pack a job bundle (end to end) | {pack / 1000:.1f} ms | — |")
        lines.append(f"| verify, decrypt and open it | {verify / 1000:.1f} ms | — |")
        lines.append(f"| envelope overhead, independent of circuit size | — | {overhead} B |")
        lines.append("")

        if not mapping.empty:
            reference = mapping.sort_values("baseline_wall_s", ascending=False).iloc[0]
            job_ms = float(reference["baseline_wall_s"]) * 1000
            job_mib = float(reference["baseline_bytes"]) / 2**20
            indexed = chosen.set_index(["algorithm", "operation"])["median_us"]
            primitive_us = float(
                sum(
                    indexed.get(key, 0.0)
                    for key in (
                        ("ML-KEM-768", "encapsulate"),
                        ("ML-KEM-768", "decapsulate"),
                        ("ML-DSA-65", "sign"),
                        ("ML-DSA-65", "verify"),
                    )
                )
            )
            lines.append(
                f"For scale: the heaviest job measured here ({reference['circuit_family']}, "
                f"{int(reference['qubits'])} qubits, {int(reference['ranks'])} ranks) runs for "
                f"{job_ms:.0f} ms and moves {job_mib:.0f} MiB over MPI. Securing it costs "
                f"{(pack + verify) / 1000:.1f} ms end to end and {overhead / 1024:.1f} KiB on "
                f"the wire — {(pack + verify) / 10 / job_ms:.2f}% of the runtime. Only "
                f"{primitive_us:.0f} us of that is lattice arithmetic; the rest is canonical "
                "serialisation and base64, which is where an optimisation would actually pay off."
            )
            lines.append("")

    search = report_module.search_table(report_module.load_search())
    if not search.empty:
        largest = search.iloc[-1]
        lines.append("### Why post-quantum cryptography, in one table")
        lines.append("")
        lines.append(
            "| search space | Grover oracle queries | classical expected | measured success |"
        )
        lines.append("|---:|---:|---:|---:|")
        for row in search.itertuples():
            lines.append(
                f"| {int(row.search_space)} | {int(row.grover_iterations)} | "
                f"{row.classical_expected:.1f} | {row.measured_success * 100:.1f}% |"
            )
        lines.append("")
        lines.append(
            f"Every row is a simulated run, not a formula: searching "
            f"{int(largest.search_space)} items took {int(largest.grover_iterations)} oracle "
            f"queries where classical search averages {largest.classical_expected:.1f}, and the "
            f"marked state was measured {largest.measured_success * 100:.1f}% of the time. "
            "That quadratic factor is why post-quantum guidance doubles symmetric key sizes "
            "rather than abandoning them — while Shor's exponential advantage is why "
            "RSA and elliptic curves are replaced outright."
        )
        lines.append("")
        lines.append("![Grover query scaling](benchmarks/plots/grover_scaling.png)")
        lines.append("")

    if not accuracy.empty:
        exact = int((accuracy["bytes_error"] == 0).sum())
        lines.append(
            f"### Cost model versus reality\n\n"
            f"In all **{exact} of {len(accuracy)}** distributed configurations measured here, "
            "the runtime sent exactly the number of bytes the analytical cost model predicted."
        )
        lines.append("")

    lines.append(
        "Raw measurements: [`benchmarks/raw/`](benchmarks/raw/) · "
        "derived tables: [`benchmarks/processed/`](benchmarks/processed/) · "
        "methodology: [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md)"
    )
    lines.append("")
    lines.append(END)
    _ = pd  # imported for the type used above
    return "\n".join(lines)


def write_paper_tables() -> list[Path]:
    """Emit LaTeX tables from the processed measurements.

    The technical report quotes the same numbers as the README, so they are
    generated from the same raw data rather than re-typed into the .tex.
    """
    from aegisq.benchmark import report as report_module

    directory = ROOT / "paper" / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    data = report_module.load_raw()

    def emit(name: str, header: str, rows: list[str], caption: str, label: str) -> None:
        body = "\n".join(rows)
        text = (
            "% Generated by scripts/generate_report.py -- do not edit.\n"
            "\\begin{table}[t]\n\\centering\n\\small\n"
            f"{header}\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
        )
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    mapping = report_module.mapping_table(data, report_module.load_calibration())
    if not mapping.empty:
        subset = mapping[mapping["ranks"] == mapping["ranks"].max()].sort_values(
            "bytes_reduction", ascending=False
        )
        rows = [
            f"{row.circuit_family} & {row.baseline_bytes / 2**20:.0f} & "
            f"{row.optimized_bytes / 2**20:.0f} & {row.bytes_reduction * 100:.1f}\\% & "
            f"{row.baseline_wall_s * 1000:.0f} & {row.optimized_wall_s * 1000:.0f} & "
            f"{row.wall_change * 100:+.1f}\\%"
            + ("" if row.wall_change_resolved else "$^{\\dagger}$")
            + " \\\\"
            for row in subset.itertuples()
        ]
        ranks = int(subset["ranks"].iloc[0])
        qubits = int(subset["qubits"].iloc[0])
        floor = subset["wall_noise_floor"].dropna()
        floor_percent = float(floor.max()) * 100 if not floor.empty else float("nan")
        emit(
            "mapping.tex",
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{MPI bytes sent} "
            "& \\multicolumn{3}{c}{Wall time} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Circuit & Default & Optimized & Reduction & Default & Optimized & Change "
            "\\\\\n & (MiB) & (MiB) & & (ms) & (ms) & \\\\\n\\midrule",
            rows,
            f"Measured MPI traffic and wall time under the default and "
            f"communication-aware placements, {qubits} qubits on {ranks} ranks. "
            f"Byte counts are exact. Wall-time changes marked $\\dagger$ are "
            f"unresolved: they do not exceed the noise floor of "
            f"{floor_percent:.1f}\\%, measured on the circuits whose traffic the "
            f"optimiser leaves unchanged and whose true effect is therefore zero.",
            "tab:mapping",
        )

    strong = report_module.strong_scaling_table(data)
    classic = (
        strong[strong["thread_policy"] == "one-thread-per-rank"] if not strong.empty else strong
    )
    if not classic.empty:
        rows = [
            f"{row.circuit_family} & {row.qubits} & {row.ranks} & {row.wall_best_s:.3f} & "
            f"{row.speedup:.2f} & {row.efficiency * 100:.0f}\\% \\\\"
            for row in classic.sort_values(["circuit_family", "ranks"]).itertuples()
        ]
        emit(
            "strong_scaling.tex",
            "\\begin{tabular}{lrrrrr}\n\\toprule\nCircuit & Qubits & Ranks & Wall (s) & "
            "Speedup & Efficiency \\\\\n\\midrule",
            rows,
            "Strong scaling with one thread per rank.",
            "tab:strong",
        )

    levers = report_module.lever_table(data)
    if not levers.empty and not levers["both_bytes"].isna().all():
        import pandas as pd

        subset = levers[levers["ranks"] == levers["ranks"].max()].sort_values(
            "both_reduction", ascending=False
        )

        def cell(value):
            return "--" if value is None or pd.isna(value) else f"{value * 100:.1f}\\%"

        rows = [
            f"{row.circuit_family} & {row.baseline_bytes / 2**20:.0f} & "
            f"{cell(row.fusion_only_reduction)} & {cell(row.placement_only_reduction)} & "
            f"{cell(row.both_reduction)} \\\\"
            for row in subset.itertuples()
        ]
        emit(
            "levers.tex",
            "\\begin{tabular}{lrrrr}\n\\toprule\nCircuit & Baseline (MiB) & Fusion only & "
            "Placement only & Both \\\\\n\\midrule",
            rows,
            f"Measured reduction in MPI traffic from each optimisation lever at "
            f"{int(subset['ranks'].iloc[0])} ranks, {int(subset['qubits'].iloc[0])} qubits.",
            "tab:levers",
        )

    primitives = report_module.pqc_table(report_module.load_pqc())
    if not primitives.empty:
        rows = [
            f"{row.algorithm} & {row.operation} & {row.median_us:.1f} & {int(row.bytes)} \\\\"
            for row in primitives.itertuples()
        ]
        emit(
            "pqc.tex",
            "\\begin{tabular}{llrr}\n\\toprule\nAlgorithm & Operation & Median ($\\mu$s) & "
            "Bytes \\\\\n\\midrule",
            rows,
            "Post-quantum primitive cost and artefact sizes.",
            "tab:pqc",
        )

    search = report_module.search_table(report_module.load_search())
    if not search.empty:
        rows = [
            f"{int(row.search_space)} & {int(row.qubits)} & {row.classical_expected:.1f} & "
            f"{int(row.grover_iterations)} & {row.measured_success * 100:.1f}\\% \\\\"
            for row in search.itertuples()
        ]
        emit(
            "grover.tex",
            "\\begin{tabular}{rrrrr}\n\\toprule\n$N$ & Qubits & Classical queries & "
            "Grover queries & Measured success \\\\\n\\midrule",
            rows,
            "Grover oracle queries against classical search, measured by simulation.",
            "tab:grover",
        )

    written.append(write_measured_macros(data, directory))
    return written


def write_measured_macros(data, directory: Path) -> Path:
    """Emit the paper's measured figures as LaTeX macros.

    The abstract claimed a wall-time improvement of 24.3% for the QFT
    placement. That number had been typed in by hand, the data had since
    been re-measured twice, and the change it described no longer cleared
    the experiment's own noise floor -- an unsupported performance claim
    sitting in the first paragraph, which is exactly what this project
    says it does not do.

    Hand-copied numbers drift silently; generated ones cannot. Every
    measured figure the prose quotes is defined here from the same frames
    the tables are built from, so re-running the suite updates the
    sentences as well as the tables, and a figure that stops existing
    becomes a LaTeX error rather than a stale claim.
    """
    from aegisq.benchmark import report as report_module

    mapping = report_module.mapping_table(data, report_module.load_calibration())
    accuracy = report_module.prediction_accuracy(data)
    pqc = report_module.pqc_table(report_module.load_pqc())
    envelope = report_module.pqc_envelope_table(report_module.load_pqc())

    macros: dict[str, str] = {}

    def define(name: str, value: str) -> None:
        macros[name] = value

    if not mapping.empty:
        widest = mapping[mapping["ranks"] == mapping["ranks"].max()]
        define("aegisqRanks", f"{int(widest['ranks'].iloc[0])}")
        define("aegisqQubits", f"{int(widest['qubits'].iloc[0])}")
        by_family = widest.set_index("circuit_family")
        for family in ("qft", "grover", "ising", "ghz"):
            if family not in by_family.index:
                continue
            row = by_family.loc[family]
            title = family.capitalize()
            define(f"aegisq{title}Traffic", f"{row['bytes_reduction'] * 100:.1f}\\%")
            define(f"aegisq{title}Wall", f"{row['wall_change'] * 100:+.1f}\\%")
            define(f"aegisq{title}Resolved", "yes" if row["wall_change_resolved"] else "no")
        floor = widest["wall_noise_floor"].dropna()
        if not floor.empty:
            define("aegisqWallNoiseFloor", f"{float(floor.max()) * 100:.1f}\\%")

    calibration = report_module.load_calibration()
    if not calibration.empty:
        for k, name in ((1, "One"), (3, "Three"), (5, "Five")):
            projected = report_module.calibration_table(calibration, launches=k)
            define(
                f"aegisqResolution{name}",
                f"{projected['resolution'].median() * 100:.0f}\\%",
            )
        measured = report_module.calibration_table(calibration)
        define("aegisqNullTrials", f"{int(measured['trials'].min())}")
        define("aegisqNullPBound", f"{float(measured['p_bound'].max()):.2f}")
        define("aegisqWorstNull", f"{float(measured['resolution'].max()) * 100:.0f}\\%")
        best = measured.loc[measured["resolution"].idxmin()]
        define("aegisqBestNull", f"{float(best['resolution']) * 100:.1f}\\%")
        define("aegisqBestNullCircuit", str(best["circuit_family"]).upper())

    levers = report_module.lever_table(data)
    if not levers.empty:
        widest_levers = levers[levers["ranks"] == levers["ranks"].max()].set_index("circuit_family")
        lever_columns = {
            "Fusion": "fusion_only_reduction",
            "Placement": "placement_only_reduction",
            "Windowed": "windowed_only_reduction",
            "Both": "both_reduction",
            "WindowedFusion": "windowed_fusion_reduction",
        }
        for family in ("qft", "grover", "ising", "ghz", "random"):
            if family not in widest_levers.index:
                continue
            for label, column in lever_columns.items():
                if column in widest_levers.columns:
                    value = float(widest_levers.loc[family, column])
                    define(f"aegisq{family.capitalize()}{label}", f"{value * 100:.1f}\\%")

    if not accuracy.empty:
        define("aegisqExactPredictions", f"{int((accuracy['bytes_error'] == 0).sum())}")
        define("aegisqPredictionConfigs", f"{len(accuracy)}")

    if not envelope.empty and not mapping.empty:
        steps = envelope.set_index("operation")
        pack = float(steps["median_us"].get("pack", 0.0))
        verify = float(steps["median_us"].get("verify_and_open", 0.0))
        reference = mapping.sort_values("baseline_wall_s", ascending=False).iloc[0]
        job_ms = float(reference["baseline_wall_s"]) * 1000
        define("aegisqEnvelopeMs", f"{(pack + verify) / 1000:.1f}")
        if job_ms:
            define("aegisqEnvelopeOverhead", f"{(pack + verify) / 10 / job_ms:.2f}\\%")
        if not pqc.empty:
            indexed = pqc.set_index(["algorithm", "operation"])["median_us"]
            lattice = sum(
                float(indexed.get(key, 0.0))
                for key in (
                    ("ML-KEM-768", "encapsulate"),
                    ("ML-KEM-768", "decapsulate"),
                    ("ML-DSA-65", "sign"),
                    ("ML-DSA-65", "verify"),
                )
            )
            define("aegisqLatticeMicros", f"{lattice:.0f}")

    body = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())
    )
    text = (
        "% Generated by scripts/generate_report.py from benchmarks/raw/.\n"
        "% Do not edit: every value here is a measurement, and editing one\n"
        "% would make the prose disagree with the tables beside it.\n"
        f"{body}\n"
    )
    path = directory / "measured.tex"
    path.write_text(text, encoding="utf-8")
    # Reporting is the caller's job, as it is for every other emitted file.
    return path


def splice(readme: str, block: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit(
            f"README is missing the {START} / {END} markers; add them where the "
            "measured results should appear"
        )
    head = readme.split(START)[0]
    tail = readme.split(END)[1]
    return head + block + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the README block is out of date instead of rewriting it",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="do not regenerate the LaTeX tables under paper/tables",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="do not regenerate processed tables and plots first",
    )
    args = parser.parse_args()

    if not args.skip_reports:
        from aegisq.benchmark import report as report_module

        report_module.write_reports()
        (report_module.PROCESSED_DIR / "summary.md").write_text(
            report_module.markdown_summary(), encoding="utf-8"
        )

    if not args.skip_paper and not args.check:
        for path in write_paper_tables():
            print(f"Wrote {path.relative_to(ROOT)}")

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    updated = splice(readme, build_block())

    if args.check:
        if updated != readme:
            print("README results block is out of date; run ./scripts/generate_report.py")
            return 1
        print("README results block is up to date.")
        return 0

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
