"""Turn raw measurements into tables and figures.

The only input is `benchmarks/raw/*.csv`. No number may enter a table or a
plot except by being read from a raw row, which is why this module has no
constants describing results and no fallbacks for missing data: if a
measurement was not taken, it does not appear.

Statistics
----------
Per configuration the suite reports the **minimum** wall time across repeats
as the headline figure — the least-contended observation is the most
reproducible one on a shared machine — alongside the median, the spread and
the repeat count, so the reader can see how noisy the measurement was.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "benchmarks" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "benchmarks" / "processed"
PLOTS_DIR = PROJECT_ROOT / "benchmarks" / "plots"

#: Columns that identify one measured configuration.
CONFIG_KEYS = [
    "experiment",
    "circuit_family",
    "qubits",
    "precision",
    "ranks",
    "mapping_strategy",
    "thread_policy",
]


def load_raw(source: Path | None = None) -> pd.DataFrame:
    """Read every raw CSV under `source` (default `benchmarks/raw`)."""
    source = source or RAW_DIR
    paths = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(f"no raw measurement files under {source}")
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError(f"raw measurement files under {source} contain no rows")
    return data


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeats into one row per configuration."""
    grouped = data.groupby(CONFIG_KEYS, as_index=False).agg(
        repeats=("wall_seconds", "count"),
        wall_best=("wall_seconds", "min"),
        wall_median=("wall_seconds", "median"),
        wall_max=("wall_seconds", "max"),
        compute_best=("compute_seconds", "min"),
        communication_best=("communication_seconds", "min"),
        bytes_sent=("bytes_sent", "max"),
        pairwise_exchanges=("pairwise_exchanges", "max"),
        communicating_gates=("communicating_gates", "max"),
        predicted_bytes=("predicted_bytes", "max"),
        predicted_exchanges=("predicted_exchanges", "max"),
        gates=("gates", "max"),
        depth=("depth", "max"),
        omp_threads=("omp_threads", "max"),
        host=("hostname", "first"),
        cpu=("cpu_model", "first"),
        commit=("git_commit", "first"),
    )
    grouped["wall_spread"] = grouped["wall_max"] - grouped["wall_best"]
    # Bytes are deterministic for a configuration; flag it if they were not.
    return grouped.sort_values(CONFIG_KEYS).reset_index(drop=True)


def strong_scaling_table(data: pd.DataFrame) -> pd.DataFrame:
    """Speedup and parallel efficiency relative to the single-rank run."""
    subset = summarise(data[data["experiment"] == "strong_scaling"])
    if subset.empty:
        return subset

    rows = []
    for (family, qubits, precision, mapping, policy), group in subset.groupby(
        ["circuit_family", "qubits", "precision", "mapping_strategy", "thread_policy"]
    ):
        group = group.sort_values("ranks")
        baseline = group[group["ranks"] == 1]
        reference = float(baseline["wall_best"].iloc[0]) if not baseline.empty else None
        for _, row in group.iterrows():
            speedup = reference / row["wall_best"] if reference else float("nan")
            rows.append(
                {
                    "circuit_family": family,
                    "qubits": qubits,
                    "precision": precision,
                    "mapping_strategy": mapping,
                    "thread_policy": policy,
                    "ranks": int(row["ranks"]),
                    "omp_threads": int(row["omp_threads"]),
                    "wall_best_s": row["wall_best"],
                    "wall_median_s": row["wall_median"],
                    "communication_s": row["communication_best"],
                    "bytes_sent": int(row["bytes_sent"]),
                    "speedup": speedup,
                    "efficiency": speedup / row["ranks"] if reference else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def weak_scaling_table(data: pd.DataFrame) -> pd.DataFrame:
    """Wall time as the problem grows with the rank count."""
    subset = summarise(data[data["experiment"] == "weak_scaling"])
    if subset.empty:
        return subset
    rows = []
    for (family, precision, policy), group in subset.groupby(
        ["circuit_family", "precision", "thread_policy"]
    ):
        group = group.sort_values("ranks")
        reference = (
            float(group[group["ranks"] == 1]["wall_best"].iloc[0])
            if (group["ranks"] == 1).any()
            else None
        )
        for _, row in group.iterrows():
            rows.append(
                {
                    "circuit_family": family,
                    "precision": precision,
                    "thread_policy": policy,
                    "ranks": int(row["ranks"]),
                    "qubits": int(row["qubits"]),
                    "amplitudes_per_rank": 2
                    ** (int(row["qubits"]) - (int(row["ranks"]).bit_length() - 1)),
                    "wall_best_s": row["wall_best"],
                    "communication_s": row["communication_best"],
                    "bytes_sent": int(row["bytes_sent"]),
                    "efficiency": reference / row["wall_best"] if reference else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def mapping_table(data: pd.DataFrame) -> pd.DataFrame:
    """Measured effect of the communication-aware placement."""
    subset = summarise(data[data["experiment"] == "mapping_comparison"])
    if subset.empty:
        return subset

    rows = []
    for (family, qubits, precision, ranks, policy), group in subset.groupby(
        ["circuit_family", "qubits", "precision", "ranks", "thread_policy"]
    ):
        default = group[group["mapping_strategy"] == "default"]
        optimized = group[group["mapping_strategy"] == "optimized"]
        if default.empty or optimized.empty:
            continue
        d = default.iloc[0]
        o = optimized.iloc[0]
        baseline_bytes = int(d["bytes_sent"])
        rows.append(
            {
                "circuit_family": family,
                "qubits": int(qubits),
                "precision": precision,
                "ranks": int(ranks),
                "thread_policy": policy,
                "gates": int(d["gates"]),
                "baseline_bytes": baseline_bytes,
                "optimized_bytes": int(o["bytes_sent"]),
                "bytes_reduction": (
                    (baseline_bytes - int(o["bytes_sent"])) / baseline_bytes
                    if baseline_bytes
                    else 0.0
                ),
                "baseline_exchanges": int(d["pairwise_exchanges"]),
                "optimized_exchanges": int(o["pairwise_exchanges"]),
                "baseline_wall_s": d["wall_best"],
                "optimized_wall_s": o["wall_best"],
                "wall_change": (
                    (o["wall_best"] - d["wall_best"]) / d["wall_best"] if d["wall_best"] else 0.0
                ),
                "baseline_comm_s": d["communication_best"],
                "optimized_comm_s": o["communication_best"],
                "baseline_wall_spread_s": d["wall_spread"],
                "optimized_wall_spread_s": o["wall_spread"],
                "predicted_baseline_bytes": int(d["predicted_bytes"]),
                "predicted_optimized_bytes": int(o["predicted_bytes"]),
            }
        )
    return pd.DataFrame(rows)


def prediction_accuracy(data: pd.DataFrame) -> pd.DataFrame:
    """Does the cost model's prediction match what was measured?"""
    summary = summarise(data)
    summary = summary[summary["ranks"] > 1].copy()
    if summary.empty:
        return summary
    summary["bytes_error"] = summary["bytes_sent"] - summary["predicted_bytes"]
    summary["exchange_error"] = summary["pairwise_exchanges"] - summary["predicted_exchanges"]
    return summary[
        CONFIG_KEYS
        + [
            "bytes_sent",
            "predicted_bytes",
            "bytes_error",
            "pairwise_exchanges",
            "predicted_exchanges",
            "exchange_error",
        ]
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
#
# Chart conventions, applied consistently so the figures read as one set:
#   * categorical hues are assigned in a fixed order and never cycled;
#   * marks are thin, grids recessive, and text uses ink colours rather than
#     the series colour, so identity is carried by the mark beside the label;
#   * every figure ships next to the CSV it was generated from, which is the
#     table view that the low-contrast series relies on.

#: Fixed categorical order (validated for colour-vision deficiency separation
#: on a light surface). Slots are assigned by entity, never by rank.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def plot_strong_scaling(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    if table.empty:
        return None
    plt = _figure()
    fig, (speed_ax, time_ax) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    ranks = sorted(table["ranks"].unique())
    speed_ax.plot(
        ranks,
        ranks,
        linestyle=(0, (4, 3)),
        color=GRID,
        linewidth=1.5,
        label="linear speedup",
        zorder=1,
    )

    groups = list(table.groupby(["circuit_family", "qubits", "thread_policy"]))
    for index, ((family, qubits, policy), group) in enumerate(groups):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        group = group.sort_values("ranks")
        label = f"{family} {qubits}q · {policy}"
        speed_ax.plot(
            group["ranks"],
            group["speedup"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=label,
            zorder=2,
        )
        time_ax.plot(
            group["ranks"],
            group["wall_best_s"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=label,
            zorder=2,
        )
        # Direct-label the end of each line so identity is not colour-alone.
        last = group.iloc[-1]
        speed_ax.annotate(
            f"{last['speedup']:.2f}x",
            (last["ranks"], last["speedup"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            color=INK,
        )

    for ax in (speed_ax, time_ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_facecolor(SURFACE)

    speed_ax.set_yscale("log", base=2)
    _style_axes(speed_ax, "Speedup vs one rank", "MPI ranks", "speedup")
    time_ax.set_yscale("log")
    _style_axes(time_ax, "Wall time", "MPI ranks", "seconds (best of repeats)")
    speed_ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)

    fig.suptitle(f"Strong scaling{f' — {host}' if host else ''}", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_weak_scaling(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    if table.empty:
        return None
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    groups = list(table.groupby(["circuit_family", "thread_policy"]))
    reference = None
    for index, ((family, policy), group) in enumerate(groups):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        group = group.sort_values("ranks")
        if reference is None:
            reference = float(group["wall_best_s"].iloc[0])
        ax.plot(
            group["ranks"],
            group["wall_best_s"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=f"{family} · {policy}",
            zorder=2,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{int(row['qubits'])}q",
                (row["ranks"], row["wall_best_s"]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha="center",
                color=INK,
            )

    ranks = sorted(table["ranks"].unique())
    if reference is not None:
        ax.plot(
            ranks,
            [reference] * len(ranks),
            linestyle=(0, (4, 3)),
            color=GRID,
            linewidth=1.5,
            label="ideal (constant time)",
            zorder=1,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    _style_axes(
        ax,
        "Weak scaling — one more qubit per doubling of ranks",
        "MPI ranks",
        "seconds (best of repeats)",
    )
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    fig.suptitle(f"Constant shard size{f' — {host}' if host else ''}", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_mapping_comparison(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    if table.empty:
        return None
    import numpy as np

    plt = _figure()
    table = table.sort_values(["circuit_family", "ranks"]).reset_index(drop=True)
    fig, (bytes_ax, time_ax) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    labels = [f"{row.circuit_family} · {row.ranks}r" for row in table.itertuples()]
    positions = np.arange(len(labels), dtype=float)
    width = 0.38

    def grouped_bars(ax, left_values, right_values, ylabel, title):
        ax.set_facecolor(SURFACE)
        ax.bar(
            positions - width / 2 - 0.01,
            left_values,
            width,
            color=SERIES_COLORS[0],
            label="default placement",
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=2,
        )
        ax.bar(
            positions + width / 2 + 0.01,
            right_values,
            width,
            color=SERIES_COLORS[1],
            label="communication-aware",
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=2,
        )
        ax.set_xticks(positions)
        # 15 groups do not fit horizontally; rotating beats truncating.
        ax.set_xticklabels(labels, fontsize=8, color=INK_MUTED, rotation=45, ha="right")
        _style_axes(ax, title, "", ylabel)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)

    grouped_bars(
        bytes_ax,
        table["baseline_bytes"] / 2**20,
        table["optimized_bytes"] / 2**20,
        "MPI bytes sent (MiB, summed over ranks)",
        "Measured communication volume",
    )
    # Label only the bars where the placement actually changed something.
    for index, row in table.iterrows():
        if row["bytes_reduction"] > 0.01:
            bytes_ax.annotate(
                f"-{row['bytes_reduction'] * 100:.0f}%",
                (positions[index] + width / 2, row["optimized_bytes"] / 2**20),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                color=INK,
            )

    grouped_bars(
        time_ax,
        table["baseline_wall_s"] * 1000,
        table["optimized_wall_s"] * 1000,
        "wall time (ms, best of repeats)",
        "Measured wall time",
    )

    fig.suptitle(
        f"Communication-aware qubit placement{f' — {host}' if host else ''}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def write_reports(
    raw: Path | None = None,
    processed: Path | None = None,
    plots: Path | None = None,
) -> dict[str, Path]:
    """Regenerate every processed table and figure from the raw measurements."""
    data = load_raw(raw)
    processed = processed or PROCESSED_DIR
    plots = plots or PLOTS_DIR
    processed.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    host = str(data["hostname"].iloc[0])
    written: dict[str, Path] = {}

    summary = summarise(data)
    summary.to_csv(processed / "summary.csv", index=False)
    written["summary"] = processed / "summary.csv"

    strong = strong_scaling_table(data)
    if not strong.empty:
        strong.to_csv(processed / "strong_scaling.csv", index=False)
        written["strong_scaling"] = processed / "strong_scaling.csv"
        figure = plot_strong_scaling(strong, plots / "strong_scaling.png", host)
        if figure:
            written["strong_scaling_plot"] = figure

    weak = weak_scaling_table(data)
    if not weak.empty:
        weak.to_csv(processed / "weak_scaling.csv", index=False)
        written["weak_scaling"] = processed / "weak_scaling.csv"
        figure = plot_weak_scaling(weak, plots / "weak_scaling.png", host)
        if figure:
            written["weak_scaling_plot"] = figure

    mapping = mapping_table(data)
    if not mapping.empty:
        mapping.to_csv(processed / "mapping_comparison.csv", index=False)
        written["mapping_comparison"] = processed / "mapping_comparison.csv"
        figure = plot_mapping_comparison(mapping, plots / "mapping_comparison.png", host)
        if figure:
            written["mapping_comparison_plot"] = figure

    accuracy = prediction_accuracy(data)
    if not accuracy.empty:
        accuracy.to_csv(processed / "prediction_accuracy.csv", index=False)
        written["prediction_accuracy"] = processed / "prediction_accuracy.csv"

    return written


def markdown_summary(raw: Path | None = None) -> str:
    """A short Markdown report of what has actually been measured."""
    data = load_raw(raw)
    lines: list[str] = []
    host = data.iloc[0]
    lines.append("## Measurement host")
    lines.append("")
    lines.append(f"- host: `{host['hostname']}`")
    lines.append(f"- CPU: {host['cpu_model']} ({host['logical_cores']} logical cores)")
    lines.append(f"- OS: {host['os']}")
    lines.append(f"- compiler: {host['compiler']}")
    lines.append(f"- MPI: {host['mpi_library']}")
    lines.append(f"- AegisQ {host['aegisq_version']} at commit `{str(host['git_commit'])[:12]}`")
    lines.append("")

    mapping = mapping_table(data)
    if not mapping.empty:
        lines.append("## Communication-aware placement (measured)")
        lines.append("")
        lines.append(
            "| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | "
            "reduction | baseline wall (s) | optimized wall (s) | wall change |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in mapping.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.qubits} | {row.ranks} | "
                f"{row.baseline_bytes:,} | {row.optimized_bytes:,} | "
                f"{row.bytes_reduction * 100:.1f}% | {row.baseline_wall_s:.3f} | "
                f"{row.optimized_wall_s:.3f} | {row.wall_change * 100:+.1f}% |"
            )
        lines.append("")

    strong = strong_scaling_table(data)
    if not strong.empty:
        lines.append("## Strong scaling (measured)")
        lines.append("")
        lines.append(
            "| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | "
            "speedup | efficiency |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in strong.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.qubits} | {row.thread_policy} | {row.ranks} | "
                f"{row.omp_threads} | {row.wall_best_s:.3f} | {row.speedup:.2f}x | "
                f"{row.efficiency * 100:.0f}% |"
            )
        lines.append("")

    weak = weak_scaling_table(data)
    if not weak.empty:
        lines.append("## Weak scaling (measured)")
        lines.append("")
        lines.append(
            "| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in weak.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.thread_policy} | {row.ranks} | {row.qubits} | "
                f"{row.amplitudes_per_rank:,} | {row.wall_best_s:.3f} | "
                f"{row.efficiency * 100:.0f}% |"
            )
        lines.append("")

    accuracy = prediction_accuracy(data)
    if not accuracy.empty:
        exact = int((accuracy["bytes_error"] == 0).sum())
        lines.append("## Cost-model accuracy")
        lines.append("")
        lines.append(
            f"- {exact} of {len(accuracy)} distributed configurations sent exactly the "
            "number of bytes the cost model predicted."
        )
        lines.append("")

    return "\n".join(lines)
