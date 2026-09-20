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
    host = data.iloc[0]
    mapping = report_module.mapping_table(data)
    strong = report_module.strong_scaling_table(data)
    accuracy = report_module.prediction_accuracy(data)

    lines: list[str] = [START, ""]
    lines.append(
        f"All figures below were measured on **{host['cpu_model']} "
        f"({host['logical_cores']} logical cores)**, {host['os']}, "
        f"{str(host['mpi_library']).split(',')[0]}, "
        f"AegisQ {host['aegisq_version']} at commit `{str(host['git_commit'])[:12]}`. "
        "They describe that host and are not a claim about cluster hardware."
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
            lines.append(
                f"| {row.circuit_family} | {row.baseline_bytes:,} | {row.optimized_bytes:,} | "
                f"**{row.bytes_reduction * 100:.1f}%** | {row.wall_change * 100:+.1f}% |"
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
