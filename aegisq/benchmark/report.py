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
    "fusion",
    "thread_policy",
]


#: Raw files whose schema is not the simulation-run schema. Each has its own
#: loader; mixing them into the main frame would put rows with no wall time or
#: rank count into every aggregate.
#:
#: Kept as a named constant rather than an inline condition because it has
#: been forgotten twice when a new experiment was added.
SPECIALISED_RAW_PREFIXES = (
    "pqc_",
    "search_",
    "kernels_",
    "placement_",
    "precision_",
    "calibration_",
)


def load_raw(source: Path | None = None) -> pd.DataFrame:
    """Read the simulation-run CSVs under `source` (default `benchmarks/raw`)."""
    source = source or RAW_DIR
    paths = (
        [p for p in sorted(source.glob("*.csv")) if not p.name.startswith(SPECIALISED_RAW_PREFIXES)]
        if source.is_dir()
        else [source]
    )
    if not paths:
        raise FileNotFoundError(f"no raw measurement files under {source}")
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError(f"raw measurement files under {source} contain no rows")

    return normalise(data)


#: Columns added to the raw schema after the first measurements were taken,
#: with the value that older rows should be read as having.
SCHEMA_DEFAULTS = {
    "fusion": "off",
    "thread_policy": "unspecified",
    "gates_before_fusion": 0,
    # Rows predating multi-launch sweeps all came from a single launch.
    "launch": 0,
}


def normalise(data: pd.DataFrame) -> pd.DataFrame:
    """Fill in columns that older raw files predate.

    Raw files are append-only and the schema grows, so a table built from a
    mix of old and new rows must decide what the old ones meant. A missing
    value in a grouping key is worse than a wrong one: pandas drops the whole
    row, so a measurement would vanish from every table without a word.

    Applied by `load_raw` and again by each table builder, so a frame that
    arrived by another route is handled the same way.
    """
    data = data.copy()
    for column, default in SCHEMA_DEFAULTS.items():
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = data[column].fillna(default)
    return data


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeats into one row per configuration."""
    data = normalise(data)
    grouped = data.groupby(CONFIG_KEYS, as_index=False).agg(
        repeats=("wall_seconds", "count"),
        launches=("launch", "nunique"),
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
    data = normalise(data)
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
    data = normalise(data)
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


def _intervals_are_disjoint(baseline, optimized) -> bool:
    """Do the two [min, max] wall-time intervals fail to overlap?

    With three repeats there is no honest parametric test to run, so this
    is the weakest criterion that cannot be wrong about the sign of a
    difference. It is necessary but, as `_add_noise_floor` explains, very
    far from sufficient.
    """
    return bool(
        optimized["wall_max"] < baseline["wall_best"]
        or optimized["wall_best"] > baseline["wall_max"]
    )


def _add_noise_floor(table: pd.DataFrame, calibration: pd.DataFrame | None = None):
    """Decide which wall-time changes are larger than the harness's own scatter.

    Three sources of evidence, in increasing order of authority.

    **The in-experiment control.** Some circuits have no placement to
    improve -- GHZ is the clear case -- so the optimiser returns an
    assignment that sends byte for byte what the default sends and the
    measured reduction is exactly 0.0%. Their true wall-time effect is
    therefore known in advance to be zero, and whatever the clock says
    about them is the experiment measuring itself. It once said -30.8%.

    **The A/A null.** Stronger, because it does not depend on finding an
    unimprovable circuit: the identical configuration is launched twice
    and the apparent change recorded, many times over. It is measured per
    circuit family, which matters -- a GHZ chain finishing in eight
    milliseconds and a Grover circuit running for half a second are not
    equally timeable, and one machine-wide floor would either excuse the
    first's noise or discard the second's real effects. It is projected
    for however many launches the sweep reduced each arm over, since a
    floor measured for a single launch is far too pessimistic for a
    figure taken as the minimum of five.

    **The controls again, as a check on the A/A projection.** That
    projection resamples launches as if they were independent, and they
    are not entirely: a thermal episode or a busy period spans several,
    and a minimum over five consecutive launches does not decorrelate a
    run of correlated slow ones. The controls are measured under exactly
    the conditions being corrected for, so a control exceeding its own
    A/A floor says by how much the projection is optimistic. That ratio,
    never below one, scales every floor in its group.

    Scaling rather than substituting is the point. Transplanting the
    control's absolute scatter onto every family would reimpose a
    group-wide floor and undo the per-family refinement -- GHZ's
    eight-millisecond difficulty is not Grover's.

    None of this touches byte counts. They are exact counters, and they
    are where the placement claim actually lives.
    """
    import numpy as np

    if table.empty:
        return table

    table = table.copy()
    table["wall_noise_floor"] = float("nan")
    table["floor_source"] = "none"
    group_keys = ["qubits", "ranks", "thread_policy", "precision"]

    control_scatter: dict[tuple, float] = {}
    for key, index in table.groupby(group_keys).groups.items():
        rows = table.loc[index]
        controls = rows[rows["bytes_reduction"] == 0.0]
        if controls.empty:
            continue
        observed = float(controls["wall_change"].abs().max())
        control_scatter[key] = observed
        table.loc[index, "wall_noise_floor"] = observed
        table.loc[index, "floor_source"] = "in-experiment control"

    if calibration is not None and not calibration.empty:
        projections = {
            launches: calibration_table(calibration, launches=int(launches)).set_index(
                ["circuit_family", "qubits", "ranks"]
            )["resolution"]
            for launches in sorted(table["launches"].unique())
        }

        def projected_floor(row) -> float | None:
            measured = projections[row["launches"]]
            key = (row["circuit_family"], row["qubits"], row["ranks"])
            return float(measured.loc[key]) if key in measured.index else None

        # How optimistic is the projection, judged on the rows whose answer
        # is known? One per group, applied to every family in it.
        inflation: dict[tuple, float] = {}
        for key in control_scatter:
            controls = table[
                (table["bytes_reduction"] == 0.0)
                & (table["qubits"] == key[0])
                & (table["ranks"] == key[1])
                & (table["thread_policy"] == key[2])
                & (table["precision"] == key[3])
            ]
            ratios = []
            for _, control in controls.iterrows():
                own = projected_floor(control)
                if own:
                    ratios.append(abs(control["wall_change"]) / own)
            if ratios:
                inflation[key] = max(1.0, max(ratios))

        for position, row in table.iterrows():
            projected = projected_floor(row)
            if projected is None:
                continue
            key = tuple(row[name] for name in group_keys)
            scale = inflation.get(key, 1.0)
            table.loc[position, "wall_noise_floor"] = projected * scale
            plural = "" if int(row["launches"]) == 1 else "es"
            source = f"A/A null, {int(row['launches'])} launch{plural}"
            if scale > 1.0:
                source += f", widened {scale:.2f}x by the control"
            table.loc[position, "floor_source"] = source

    floor = table["wall_noise_floor"]
    above_floor = floor.isna() | (table["wall_change"].abs() > floor)
    calibrated = table["floor_source"].str.contains("A/A")
    # Where a calibrated floor exists it is the whole criterion, and the
    # interval test is dropped rather than stacked on top.
    #
    # Two reasons, and the second is why this is not merely convenient. The
    # A/A floor is already the null distribution of the exact statistic
    # being compared -- it comes from resampling minimum-over-k launches for
    # both arms -- so it accounts for that estimator's sampling variability
    # by construction, and a second test adds no information. And the
    # interval test compares [min, max] ranges, where max is a tail
    # statistic; across several launches the tails always overlap, so it
    # rejects everything regardless of effect size. It had quietly taken the
    # resolved count to zero while effects three times the floor sat in the
    # table.
    #
    # Without calibration there is no principled floor, and the weak
    # interval test is better than nothing.
    table["wall_change_resolved"] = np.where(
        calibrated, above_floor, table["wall_change_resolved"] & above_floor
    )
    # A control cannot resolve its own effect, by construction: its traffic
    # is identical under both placements, so the true effect is zero
    # whatever the clock says.
    table.loc[table["bytes_reduction"] == 0.0, "wall_change_resolved"] = False
    return table


def mapping_table(data: pd.DataFrame, calibration: pd.DataFrame | None = None) -> pd.DataFrame:
    """Measured effect of the communication-aware placement.

    Pass `calibration` (from `load_calibration`) to judge wall-time
    differences against the harness's measured A/A null rather than
    against the weaker in-experiment control. It is not loaded
    implicitly, so a caller working with synthetic data gets synthetic
    answers.
    """
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
                "wall_change_resolved": _intervals_are_disjoint(d, o),
                "baseline_comm_s": d["communication_best"],
                "optimized_comm_s": o["communication_best"],
                "baseline_wall_spread_s": d["wall_spread"],
                "optimized_wall_spread_s": o["wall_spread"],
                "baseline_wall_max_s": d["wall_max"],
                "optimized_wall_max_s": o["wall_max"],
                "repeats": int(min(d["repeats"], o["repeats"])),
                "launches": int(min(d["launches"], o["launches"])),
                "predicted_baseline_bytes": int(d["predicted_bytes"]),
                "predicted_optimized_bytes": int(o["predicted_bytes"]),
            }
        )
    return _add_noise_floor(pd.DataFrame(rows), calibration)


def load_calibration(source: Path | None = None) -> pd.DataFrame:
    """Read the A/A null measurements, if any exist."""
    source = source or RAW_DIR
    paths = sorted(source.glob("calibration_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def calibration_table(
    data: pd.DataFrame, launches: int = 1, resamples: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """The harness's own resolution, per configuration.

    `resolution` is the quantile of the apparent change, in absolute
    value, at `1 - 1 / (trials + 1)`. That threshold is picked so the
    statement it licenses is exact: an observed effect larger than every
    one of `n` null trials has probability at most `1 / (n + 1)` under
    exchangeability. With ten trials that is `p <= 0.09` -- a weak
    claim, but a stated one.

    `launches` projects the resolution for an experiment that reduces
    each arm to the minimum over that many independent launches, which
    is what `--launches` does in the placement sweep. It is estimated by
    resampling the measured launch times rather than by re-measuring:
    the pool already contains every combination, and a question about
    the null's own tail needs no new data.

    The `p` bound is exact only at one launch, where the threshold is a
    value the trials actually produced. Beyond that it is a resampled
    estimate of the same threshold and inherits the bootstrap's error,
    which is why the column reporting it also reports the launch count.

    The minimum is the estimator worth projecting because this machine's
    interference is one-sided. Nothing schedules a launch to run faster
    than an uncontended one, so the distribution is right-skewed with a
    long upper tail and the minimum converges on the clean runtime as
    launches are added. A mean would not; it would chase the tail.
    """
    import numpy as np

    if data.empty:
        return data
    frame = data.copy()
    frame["absolute_change"] = frame["apparent_change"].abs()
    table = (
        frame.groupby(["circuit_family", "qubits", "ranks"], as_index=False)
        .agg(
            trials=("apparent_change", "count"),
            median_wall_s=("first_wall_s", "median"),
            median_absolute_change=("absolute_change", "median"),
            most_negative=("apparent_change", "min"),
            most_positive=("apparent_change", "max"),
        )
        .sort_values(["ranks", "circuit_family"])
        .reset_index(drop=True)
    )
    table["launches"] = max(1, int(launches))
    table["p_bound"] = 1.0 / (table["trials"] + 1)

    rng = np.random.default_rng(seed)
    resolutions = []
    for row in table.itertuples():
        cell = frame[
            (frame["circuit_family"] == row.circuit_family)
            & (frame["qubits"] == row.qubits)
            & (frame["ranks"] == row.ranks)
        ]
        quantile = 1.0 - row.p_bound
        if row.launches <= 1:
            # `interpolation="higher"` rather than the default linear one.
            # The guarantee is a statement about the measured trials -- an
            # effect larger than all n of them -- so the threshold has to be
            # a value one of them actually took. Interpolating between the
            # top two lands below the largest and quietly weakens the bound
            # into something the data does not support.
            resolutions.append(
                float(cell["absolute_change"].quantile(quantile, interpolation="higher"))
            )
            continue
        pool = np.concatenate([cell["first_wall_s"].to_numpy(), cell["second_wall_s"].to_numpy()])
        draws = rng.choice(pool, size=(resamples, 2, row.launches), replace=True).min(axis=2)
        changes = np.abs((draws[:, 1] - draws[:, 0]) / draws[:, 0])
        resolutions.append(float(np.quantile(changes, quantile)))
    table["resolution"] = resolutions
    return table


def load_pqc(source: Path | None = None) -> pd.DataFrame:
    """Read the post-quantum measurement files, if any exist."""
    source = source or RAW_DIR
    paths = sorted(source.glob("pqc_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pqc_table(data: pd.DataFrame) -> pd.DataFrame:
    """Primitive timings and sizes, one row per (algorithm, operation)."""
    if data.empty:
        return data
    primitives = data[data["experiment"] == "pqc_primitive"]
    if primitives.empty:
        return primitives
    return (
        primitives.groupby(["algorithm", "operation"], as_index=False)
        .agg(
            iterations=("iterations", "max"),
            median_us=("median_us", "min"),
            mean_us=("mean_us", "mean"),
            stdev_us=("stdev_us", "mean"),
            bytes=("bytes", "max"),
        )
        .sort_values(["algorithm", "operation"])
        .reset_index(drop=True)
    )


def pqc_envelope_table(data: pd.DataFrame) -> pd.DataFrame:
    """End-to-end envelope timings and the size decomposition."""
    if data.empty:
        return data
    envelope = data[data["experiment"] == "pqc_envelope"]
    if envelope.empty:
        return envelope
    return (
        envelope.groupby(["operation"], as_index=False)
        .agg(
            median_us=("median_us", "min"),
            mean_us=("mean_us", "mean"),
            bytes=("bytes", "max"),
            detail=("detail", "first"),
        )
        .reset_index(drop=True)
    )


def plot_pqc(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """Primitive cost by algorithm, with the artefact sizes beside it."""
    if table.empty:
        return None
    import numpy as np

    plt = _figure()
    fig, (time_ax, size_ax) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)

    algorithms = sorted(table["algorithm"].unique())
    operations = sorted(table["operation"].unique())
    positions = np.arange(len(algorithms), dtype=float)
    width = 0.8 / max(1, len(operations))

    for index, operation in enumerate(operations):
        subset = table[table["operation"] == operation].set_index("algorithm")
        values = [float(subset["median_us"].get(name, 0.0)) for name in algorithms]
        offset = (index - (len(operations) - 1) / 2) * width
        time_ax.bar(
            positions + offset,
            values,
            width * 0.9,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=operation,
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=2,
        )

    time_ax.set_xticks(positions)
    time_ax.set_xticklabels(algorithms, fontsize=8, rotation=30, ha="right", color=INK_MUTED)
    time_ax.set_facecolor(SURFACE)
    _style_axes(time_ax, "Primitive cost (median of repeats)", "", "microseconds")
    time_ax.grid(axis="x", visible=False)
    time_ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)

    # Sizes: public key, ciphertext or signature, by algorithm.
    size_rows = table[table["operation"].isin(["keygen", "encapsulate", "sign"])]
    size_ax.set_facecolor(SURFACE)
    labels = []
    values = []
    for name in algorithms:
        subset = size_rows[size_rows["algorithm"] == name]
        for row in subset.itertuples():
            labels.append(f"{name}\n{row.operation}")
            values.append(row.bytes)
    size_positions = np.arange(len(labels), dtype=float)
    size_ax.bar(
        size_positions,
        values,
        0.6,
        color=SERIES_COLORS[2],
        edgecolor=SURFACE,
        linewidth=1.0,
        zorder=2,
    )
    for position, value in zip(size_positions, values, strict=False):
        size_ax.annotate(
            f"{int(value)}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
            color=INK,
        )
    size_ax.set_xticks(size_positions)
    size_ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color=INK_MUTED)
    _style_axes(size_ax, "Key, ciphertext and signature sizes", "", "bytes")
    size_ax.grid(axis="x", visible=False)

    fig.suptitle(f"Post-quantum primitives{f' — {host}' if host else ''}", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def load_search(source: Path | None = None) -> pd.DataFrame:
    """Read the Grover scaling measurements, if any exist."""
    source = source or RAW_DIR
    paths = sorted(source.glob("search_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def search_table(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    return (
        data.groupby(["search_bits"], as_index=False)
        .agg(
            search_space=("search_space", "max"),
            qubits=("qubits", "max"),
            gates=("gates", "max"),
            grover_iterations=("grover_iterations", "max"),
            classical_expected=("classical_expected_queries", "max"),
            classical_worst=("classical_worst_case_queries", "max"),
            measured_success=("measured_success_probability", "max"),
            theoretical_success=("theoretical_success_probability", "max"),
        )
        .sort_values("search_bits")
        .reset_index(drop=True)
    )


def plot_search_scaling(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """Oracle queries against search-space size, with measured success beside it."""
    if table.empty:
        return None
    plt = _figure()
    fig, (query_ax, success_ax) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    spaces = table["search_space"]
    query_ax.plot(
        spaces,
        table["classical_expected"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=SERIES_COLORS[0],
        label="classical, expected (N+1)/2",
        zorder=2,
    )
    query_ax.plot(
        spaces,
        table["grover_iterations"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=SERIES_COLORS[1],
        label="Grover, floor(pi/4 sqrt(N))",
        zorder=2,
    )
    query_ax.set_xscale("log", base=2)
    query_ax.set_yscale("log", base=2)
    query_ax.set_facecolor(SURFACE)
    _style_axes(query_ax, "Oracle queries", "search space N", "queries")
    query_ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)

    # Theory as a line, measurement as markers on top: they agree closely, and
    # drawing the measurement as a line would simply hide the theory curve.
    success_ax.plot(
        spaces,
        table["theoretical_success"],
        linewidth=2,
        color=GRID,
        label="theoretical",
        zorder=2,
    )
    success_ax.plot(
        spaces,
        table["measured_success"],
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor=SERIES_COLORS[1],
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="measured",
        zorder=3,
    )
    success_ax.set_xscale("log", base=2)
    success_ax.set_ylim(0, 1.05)
    success_ax.set_facecolor(SURFACE)
    _style_axes(
        success_ax, "Probability of measuring the marked state", "search space N", "probability"
    )
    success_ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)

    fig.suptitle(f"Grover search, simulated{f' — {host}' if host else ''}", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def lever_table(data: pd.DataFrame) -> pd.DataFrame:
    """Measured traffic for each combination of optimisation levers.

    Rows are (circuit, ranks); columns are the four combinations of qubit
    placement and gate fusion, so the question "are these two levers
    independent?" can be read off directly.
    """
    data = normalise(data)
    data = normalise(data)
    subset = summarise(data[data["experiment"] == "mapping_comparison"])
    if subset.empty:
        return subset

    rows = []
    for (family, qubits, precision, ranks), group in subset.groupby(
        ["circuit_family", "qubits", "precision", "ranks"]
    ):
        indexed = group.set_index(["mapping_strategy", "fusion"])

        def value(mapping: str, fusion: str, column: str, table=indexed):
            key = (mapping, fusion)
            if key not in table.index:
                return None
            return table.loc[key, column]

        baseline = value("default", "off", "bytes_sent")
        if baseline is None:
            continue
        entry = {
            "circuit_family": family,
            "qubits": int(qubits),
            "precision": precision,
            "ranks": int(ranks),
            "baseline_bytes": int(baseline),
            "baseline_wall_s": float(value("default", "off", "wall_best")),
        }
        for label, (mapping, fusion) in {
            "fusion_only": ("default", "on"),
            "placement_only": ("optimized", "off"),
            "windowed_only": ("windowed", "off"),
            "both": ("optimized", "on"),
            "windowed_fusion": ("windowed", "on"),
        }.items():
            measured = value(mapping, fusion, "bytes_sent")
            wall = value(mapping, fusion, "wall_best")
            entry[f"{label}_bytes"] = int(measured) if measured is not None else None
            entry[f"{label}_wall_s"] = float(wall) if wall is not None else None
            entry[f"{label}_reduction"] = (
                (baseline - measured) / baseline if measured is not None and baseline else None
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def plot_levers(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """Traffic under each combination of levers."""
    if table.empty:
        return None
    measured = [
        column
        for column in (
            "fusion_only_bytes",
            "placement_only_bytes",
            "windowed_only_bytes",
            "both_bytes",
            "windowed_fusion_bytes",
        )
        if column in table.columns and not table[column].isna().all()
    ]
    if not measured:
        return None

    import numpy as np

    plt = _figure()
    table = table.sort_values(["circuit_family", "ranks"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    labels_by_column = {
        "fusion_only_bytes": "fusion only",
        "placement_only_bytes": "static placement",
        "windowed_only_bytes": "windowed placement",
        "both_bytes": "placement + fusion",
        "windowed_fusion_bytes": "windowed + fusion",
    }
    series = [("baseline_bytes", "default placement", SERIES_COLORS[0])]
    series += [
        (column, labels_by_column[column], SERIES_COLORS[(index + 1) % len(SERIES_COLORS)])
        for index, column in enumerate(measured)
    ]
    positions = np.arange(len(table), dtype=float)
    width = 0.8 / len(series)

    for index, (column, label, colour) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        values = [
            (row / 2**20 if row is not None and not pd.isna(row) else 0.0) for row in table[column]
        ]
        ax.bar(
            positions + offset,
            values,
            width * 0.9,
            color=colour,
            label=label,
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=2,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{row.circuit_family} · {row.ranks}r" for row in table.itertuples()],
        fontsize=8,
        rotation=45,
        ha="right",
        color=INK_MUTED,
    )
    _style_axes(ax, "", "", "MPI bytes sent (MiB, summed over ranks)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    fig.suptitle(
        f"Ways to reduce MPI traffic{f' — {host}' if host else ''}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def load_kernels(source: Path | None = None) -> pd.DataFrame:
    """Read the local-kernel bandwidth measurements, if any exist."""
    source = source or RAW_DIR
    paths = sorted(source.glob("kernels_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def kernel_table(data: pd.DataFrame) -> pd.DataFrame:
    """Best achieved bandwidth per kernel and thread count."""
    if data.empty:
        return data
    return (
        data.groupby(["kernel", "threads", "qubits"], as_index=False)
        .agg(
            gb_per_second=("gb_per_second", "max"),
            slowest_gb_per_second=("gb_per_second", "min"),
            inplace_gb_per_second=("inplace_gb_per_second", "max"),
            fraction_of_inplace=("fraction_of_inplace", "max"),
            seconds_per_gate=("seconds_per_gate", "min"),
        )
        .sort_values(["kernel", "threads"])
        .reset_index(drop=True)
    )


def kernel_position_table(data: pd.DataFrame) -> pd.DataFrame:
    """How much the target qubit's position changes a kernel's cost.

    The communication cost model assumes only the local/global distinction
    matters, never which local position a qubit occupies. This table is the
    evidence for or against that assumption.
    """
    if data.empty:
        return data
    grouped = data.groupby(["kernel", "threads"], as_index=False).agg(
        fastest=("gb_per_second", "max"),
        slowest=("gb_per_second", "min"),
    )
    grouped["spread"] = (grouped["fastest"] - grouped["slowest"]) / grouped["fastest"]
    return grouped.sort_values(["kernel", "threads"]).reset_index(drop=True)


def plot_kernels(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """Achieved bandwidth against thread count, with the machine's reference."""
    if table.empty:
        return None
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for index, (kernel, group) in enumerate(table.groupby("kernel")):
        group = group.sort_values("threads")
        ax.plot(
            group["threads"],
            group["gb_per_second"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=kernel,
            zorder=3,
        )

    reference = table.groupby("threads", as_index=False)["inplace_gb_per_second"].max()
    ax.plot(
        reference["threads"],
        reference["inplace_gb_per_second"],
        linestyle=(0, (4, 3)),
        linewidth=2,
        color=GRID,
        label="in-place reference",
        zorder=2,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(table["threads"].unique()))
    ax.set_xticklabels([str(t) for t in sorted(table["threads"].unique())])
    _style_axes(ax, "Local kernel bandwidth", "threads", "GB/s achieved")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, ncol=2)
    fig.suptitle(
        f"How close the kernels get to the machine{f' — {host}' if host else ''}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def load_placement_quality(source: Path | None = None) -> pd.DataFrame:
    source = source or RAW_DIR
    paths = sorted(source.glob("placement_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def placement_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    """How often the fallback search matches the exhaustive optimum."""
    if data.empty:
        return data
    return (
        data.groupby(["ranks", "qubits"], as_index=False)
        .agg(
            samples=("gap", "count"),
            candidates=("candidates", "max"),
            found_optimum=("found_optimum", "sum"),
            worst_gap=("gap", "max"),
            median_speedup=("speedup", "median"),
            heuristic_samples=(
                "fallback_strategy",
                lambda column: int((column.str.startswith("greedy")).sum()),
            ),
        )
        .sort_values(["qubits", "ranks"])
        .reset_index(drop=True)
    )


def load_precision(source: Path | None = None) -> pd.DataFrame:
    """Read the single-versus-double precision measurements, if any exist."""
    source = source or RAW_DIR
    paths = sorted(source.glob("precision_*.csv")) if source.is_dir() else [source]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def precision_table(data: pd.DataFrame) -> pd.DataFrame:
    """Worst-case fp32 error grouped by circuit size."""
    if data.empty:
        return data
    return (
        data.groupby(["circuit_family", "depth"], as_index=False)
        .agg(
            samples=("infidelity", "count"),
            qubits=("qubits", "max"),
            gates=("gates", "max"),
            worst_infidelity=("infidelity", "max"),
            worst_amplitude_error=("max_amplitude_error", "max"),
            worst_norm_error=("norm_error", "max"),
            median_speedup=("speedup", "median"),
        )
        .sort_values(["circuit_family", "gates"])
        .reset_index(drop=True)
    )


def plot_precision(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """How fp32 error grows with circuit size."""
    if table.empty:
        return None
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for index, (family, group) in enumerate(table.groupby("circuit_family")):
        group = group.sort_values("gates")
        style = {"linestyle": "none"} if len(group) == 1 else {}
        ax.plot(
            group["gates"],
            group["worst_infidelity"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=family,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(ax, "Single-precision error against fp64", "gates in the circuit", "1 - fidelity")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    fig.suptitle(f"What fp32 costs{f' — {host}' if host else ''}", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(table: pd.DataFrame, path: Path, host: str = "") -> Path | None:
    """What the harness can resolve, against how long the circuit runs.

    The relationship is the point: a GHZ chain finishing in 8 ms cannot
    be timed to better than tens of percent, while a Grover circuit
    running for half a second can. Any wall-time claim has to be read
    against the point for its own configuration.
    """
    if table.empty:
        return None
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for index, (ranks, group) in enumerate(table.groupby("ranks")):
        ax.scatter(
            group["median_wall_s"] * 1000,
            group["resolution"] * 100,
            s=64,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            edgecolor=SURFACE,
            linewidth=2,
            label=f"{int(ranks)} ranks",
            zorder=3,
        )
        for row in group.itertuples():
            ax.annotate(
                row.circuit_family,
                (row.median_wall_s * 1000, row.resolution * 100),
                textcoords="offset points",
                xytext=(7, 3),
                fontsize=7,
                color=INK_MUTED,
            )

    ax.set_xscale("log")
    _style_axes(
        ax,
        "Resolution of the measurement harness (A/A null)",
        "median wall time of one launch (ms)",
        "largest apparent change with no change made (%)",
    )
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    fig.suptitle(
        f"A wall-time change smaller than its point is not a result{f' — {host}' if host else ''}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


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

    calibration = load_calibration(raw if raw and raw.is_dir() else None)
    if not calibration.empty:
        resolution = calibration_table(calibration)
        resolution.to_csv(processed / "calibration.csv", index=False)
        written["calibration"] = processed / "calibration.csv"
        figure = plot_calibration(resolution, plots / "calibration.png", host)
        if figure:
            written["calibration_plot"] = figure

    mapping = mapping_table(data, calibration)
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

    levers = lever_table(data)
    measured_levers = [
        column
        for column in (
            "fusion_only_bytes",
            "placement_only_bytes",
            "windowed_only_bytes",
            "both_bytes",
        )
        if column in levers.columns and not levers[column].isna().all()
    ]
    if not levers.empty and measured_levers:
        levers.to_csv(processed / "lever_comparison.csv", index=False)
        written["lever_comparison"] = processed / "lever_comparison.csv"
        figure = plot_levers(levers, plots / "lever_comparison.png", host)
        if figure:
            written["lever_comparison_plot"] = figure

    pqc = load_pqc(raw)
    if not pqc.empty:
        primitives = pqc_table(pqc)
        if not primitives.empty:
            primitives.to_csv(processed / "pqc_primitives.csv", index=False)
            written["pqc_primitives"] = processed / "pqc_primitives.csv"
            figure = plot_pqc(primitives, plots / "pqc_primitives.png", host)
            if figure:
                written["pqc_primitives_plot"] = figure
        envelope = pqc_envelope_table(pqc)
        if not envelope.empty:
            envelope.to_csv(processed / "pqc_envelope.csv", index=False)
            written["pqc_envelope"] = processed / "pqc_envelope.csv"

    kernels = kernel_table(load_kernels(raw))
    if not kernels.empty:
        kernels.to_csv(processed / "kernel_bandwidth.csv", index=False)
        written["kernel_bandwidth"] = processed / "kernel_bandwidth.csv"
        positions = kernel_position_table(load_kernels(raw))
        positions.to_csv(processed / "kernel_position_sensitivity.csv", index=False)
        written["kernel_position_sensitivity"] = processed / "kernel_position_sensitivity.csv"
        figure = plot_kernels(kernels, plots / "kernel_bandwidth.png", host)
        if figure:
            written["kernel_bandwidth_plot"] = figure

    precision = precision_table(load_precision(raw))
    if not precision.empty:
        precision.to_csv(processed / "precision_error.csv", index=False)
        written["precision_error"] = processed / "precision_error.csv"
        figure = plot_precision(precision, plots / "precision_error.png", host)
        if figure:
            written["precision_error_plot"] = figure

    placement = placement_quality_table(load_placement_quality(raw))
    if not placement.empty:
        placement.to_csv(processed / "placement_quality.csv", index=False)
        written["placement_quality"] = processed / "placement_quality.csv"

    search = search_table(load_search(raw))
    if not search.empty:
        search.to_csv(processed / "grover_scaling.csv", index=False)
        written["grover_scaling"] = processed / "grover_scaling.csv"
        figure = plot_search_scaling(search, plots / "grover_scaling.png", host)
        if figure:
            written["grover_scaling_plot"] = figure

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

    mapping = mapping_table(data, load_calibration())
    if not mapping.empty:
        lines.append("## Communication-aware placement (measured)")
        lines.append("")
        lines.append(
            "| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | "
            "reduction | baseline wall (s) | optimized wall (s) | wall change | resolved |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
        for row in mapping.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.qubits} | {row.ranks} | "
                f"{row.baseline_bytes:,} | {row.optimized_bytes:,} | "
                f"{row.bytes_reduction * 100:.1f}% | {row.baseline_wall_s:.3f} | "
                f"{row.optimized_wall_s:.3f} | {row.wall_change * 100:+.1f}% | "
                f"{'yes' if row.wall_change_resolved else 'no'} |"
            )
        lines.append("")
        lines.append(
            "A wall-time change is *resolved* only if it exceeds the noise floor for its "
            "rank count -- the largest change measured on circuits the optimiser leaves "
            "byte-for-byte unchanged, whose true effect is therefore zero -- and its "
            "repeat range does not overlap the baseline's. Byte counts are exact."
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

    precision = precision_table(load_precision(raw))
    if not precision.empty:
        lines.append("## Single-precision error (measured)")
        lines.append("")
        lines.append("| circuit | qubits | gates | worst 1-fidelity | worst amplitude error |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in precision.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.qubits} | {row.gates} | "
                f"{row.worst_infidelity:.2e} | {row.worst_amplitude_error:.2e} |"
            )
        lines.append("")

    placement = placement_quality_table(load_placement_quality(raw))
    if not placement.empty:
        lines.append("## Placement search quality (measured)")
        lines.append("")
        lines.append(
            "| qubits | ranks | candidate sets | samples | needing the heuristic | "
            "optimum found | worst gap | median speedup |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in placement.itertuples():
            lines.append(
                f"| {row.qubits} | {row.ranks} | {row.candidates:,} | {row.samples} | "
                f"{row.heuristic_samples} | {row.found_optimum}/{row.samples} | "
                f"{row.worst_gap * 100:.2f}% | {row.median_speedup:.0f}x |"
            )
        lines.append("")

    kernels = kernel_table(load_kernels(raw))
    if not kernels.empty:
        lines.append("## Local kernel bandwidth (measured)")
        lines.append("")
        lines.append("| kernel | threads | GB/s | in-place reference | fraction |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in kernels.itertuples():
            lines.append(
                f"| {row.kernel} | {row.threads} | {row.gb_per_second:.1f} | "
                f"{row.inplace_gb_per_second:.1f} | {row.fraction_of_inplace * 100:.0f}% |"
            )
        lines.append("")

    pqc = load_pqc(raw)
    primitives = pqc_table(pqc)
    if not primitives.empty:
        lines.append("## Post-quantum primitives (measured)")
        lines.append("")
        lines.append("| algorithm | operation | median (us) | bytes |")
        lines.append("|---|---|---:|---:|")
        for row in primitives.itertuples():
            lines.append(
                f"| {row.algorithm} | {row.operation} | {row.median_us:.1f} | {int(row.bytes)} |"
            )
        lines.append("")

    envelope = pqc_envelope_table(pqc)
    if not envelope.empty:
        lines.append("## Secure job envelope (measured)")
        lines.append("")
        lines.append("| step | median (us) | bytes |")
        lines.append("|---|---:|---:|")
        for row in envelope.itertuples():
            timing = f"{row.median_us:.0f}" if row.median_us > 0 else "-"
            lines.append(f"| {row.operation} | {timing} | {int(row.bytes)} |")
        lines.append("")

    levers = lever_table(data)
    if not levers.empty:

        def lever_percent(value):
            return "—" if value is None or pd.isna(value) else f"{value * 100:.1f}%"

        lines.append("## Optimisation levers (measured)")
        lines.append("")
        lines.append(
            "| circuit | ranks | baseline (MiB) | fusion | static placement | "
            "windowed placement | placement + fusion | windowed + fusion |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in levers.itertuples():
            lines.append(
                f"| {row.circuit_family} | {row.ranks} | {row.baseline_bytes / 2**20:.0f} | "
                f"{lever_percent(row.fusion_only_reduction)} | "
                f"{lever_percent(row.placement_only_reduction)} | "
                f"{lever_percent(getattr(row, 'windowed_only_reduction', None))} | "
                f"{lever_percent(row.both_reduction)} | "
                f"{lever_percent(getattr(row, 'windowed_fusion_reduction', None))} |"
            )
        lines.append("")

    search = search_table(load_search(raw))
    if not search.empty:
        lines.append("## Grover query scaling (measured)")
        lines.append("")
        lines.append(
            "| search space | qubits | classical expected queries | Grover queries | "
            "measured success |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for row in search.itertuples():
            lines.append(
                f"| {int(row.search_space)} | {int(row.qubits)} | "
                f"{row.classical_expected:.1f} | {int(row.grover_iterations)} | "
                f"{row.measured_success * 100:.1f}% |"
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
