"""Phase 11: the measurement harness itself.

The suite's credibility rests on these mechanics, so they get tests: raw rows
carry their provenance, repeats are aggregated the way the methodology says,
and derived tables are computed from raw rows rather than from constants.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from aegisq.benchmark import report as report_module
from aegisq.benchmark.runner import (
    RAW_FIELDS,
    BenchmarkConfig,
    append_rows,
    environment_row,
    measure,
)
from aegisq.benchmark.scaling import resolve_threads, threads_for


def test_environment_row_carries_provenance():
    row = environment_row()
    for key in (
        "timestamp",
        "hostname",
        "cpu_model",
        "logical_cores",
        "os",
        "compiler",
        "mpi_library",
        "git_commit",
        "aegisq_version",
    ):
        assert key in row and row[key] not in (None, "")


def test_measure_returns_one_row_per_repeat_with_the_full_schema():
    config = BenchmarkConfig(circuit_family="ghz", qubits=6, repeats=2, warmup=1)
    rows = measure(config)
    assert len(rows) == 2
    for row in rows:
        assert set(RAW_FIELDS) <= set(row)
        assert row["qubits"] == 6
        assert row["gates"] > 0
        assert row["wall_seconds"] > 0
        assert row["circuit_family"] == "ghz"


def test_warmup_repeats_are_discarded():
    config = BenchmarkConfig(circuit_family="ghz", qubits=5, repeats=1, warmup=3)
    rows = measure(config)
    assert len(rows) == 1
    assert rows[0]["repeat"] == 0


def test_single_rank_measurement_records_no_traffic():
    rows = measure(BenchmarkConfig(circuit_family="qft", qubits=6, repeats=1, warmup=0))
    assert rows[0]["bytes_sent"] == 0
    assert rows[0]["predicted_bytes"] == 0


def test_append_rows_writes_the_header_once(tmp_path):
    path = tmp_path / "raw.csv"
    rows = measure(BenchmarkConfig(circuit_family="ghz", qubits=4, repeats=1, warmup=0))
    append_rows(path, rows)
    append_rows(path, rows)

    with path.open(encoding="utf-8") as handle:
        contents = list(csv.reader(handle))
    assert contents[0] == RAW_FIELDS
    assert len(contents) == 3  # header plus two rows
    assert contents[1][RAW_FIELDS.index("circuit_family")] == "ghz"


def test_append_rows_ignores_empty_input(tmp_path):
    path = tmp_path / "empty.csv"
    append_rows(path, [])
    assert not path.exists()


@pytest.mark.parametrize(
    "ranks, cores, expected",
    [(1, 8, 8), (2, 8, 4), (4, 8, 2), (8, 8, 1), (16, 8, 1)],
)
def test_fixed_total_cores_policy(ranks, cores, expected):
    assert threads_for(ranks, cores) == expected


def test_one_thread_per_rank_policy():
    for ranks in (1, 2, 4, 8):
        assert resolve_threads(ranks, "one-thread-per-rank") == 1


def test_unknown_thread_policy_is_rejected():
    with pytest.raises(ValueError, match="thread policy"):
        resolve_threads(4, "as-many-as-possible")


def synthetic_raw() -> pd.DataFrame:
    """Two configurations, two repeats each, with known numbers."""
    base = {
        "timestamp": "2026-01-01T00:00:00",
        "hostname": "test-host",
        "cpu_model": "Test CPU",
        "logical_cores": 8,
        "os": "test",
        "python_version": "3.12.0",
        "compiler": "TestClang",
        "mpi_library": "Test MPI",
        "git_commit": "abc123",
        "git_dirty": 0,
        "aegisq_version": "0.1.0",
        "circuit_family": "ising",
        "circuit_name": "ising20",
        "qubits": 20,
        "gates": 100,
        "depth": 40,
        "two_qubit_gates": 50,
        "precision": "fp64",
        "shots": 0,
        "seed": 42,
        "communicating_gates": 10,
        "local_amplitudes": 262144,
        "thread_policy": "one-thread-per-rank",
        "omp_threads": 1,
    }
    rows = []
    for ranks, wall in ((1, 4.0), (1, 4.4), (4, 1.0), (4, 1.2)):
        rows.append(
            {
                **base,
                "experiment": "strong_scaling",
                "ranks": ranks,
                "mapping_strategy": "default",
                "repeat": 0 if wall in (4.0, 1.0) else 1,
                "wall_seconds": wall,
                "compute_seconds": wall * 0.8,
                "communication_seconds": wall * 0.2,
                "bytes_sent": 0 if ranks == 1 else 1000,
                "bytes_received": 0 if ranks == 1 else 1000,
                "pairwise_exchanges": 0 if ranks == 1 else 8,
                "predicted_bytes": 0 if ranks == 1 else 1000,
                "predicted_exchanges": 0 if ranks == 1 else 8,
            }
        )
    for mapping, sent in (("default", 2000), ("optimized", 500)):
        rows.append(
            {
                **base,
                "experiment": "mapping_comparison",
                "ranks": 4,
                "mapping_strategy": mapping,
                "repeat": 0,
                "wall_seconds": 2.0 if mapping == "default" else 1.5,
                "compute_seconds": 1.0,
                "communication_seconds": 0.5,
                "bytes_sent": sent,
                "bytes_received": sent,
                "pairwise_exchanges": 8,
                "predicted_bytes": sent,
                "predicted_exchanges": 8,
            }
        )
    return pd.DataFrame(rows)


def test_summarise_uses_best_of_repeats():
    summary = report_module.summarise(synthetic_raw())
    single = summary[(summary["ranks"] == 1) & (summary["experiment"] == "strong_scaling")]
    assert single["wall_best"].iloc[0] == 4.0
    assert single["wall_median"].iloc[0] == pytest.approx(4.2)
    assert single["wall_spread"].iloc[0] == pytest.approx(0.4)
    assert single["repeats"].iloc[0] == 2


def test_strong_scaling_table_computes_speedup_and_efficiency():
    table = report_module.strong_scaling_table(synthetic_raw())
    four = table[table["ranks"] == 4].iloc[0]
    assert four["speedup"] == pytest.approx(4.0)  # 4.0 s -> 1.0 s
    assert four["efficiency"] == pytest.approx(1.0)


def test_mapping_table_reports_measured_reduction():
    table = report_module.mapping_table(synthetic_raw())
    row = table.iloc[0]
    assert row["baseline_bytes"] == 2000
    assert row["optimized_bytes"] == 500
    assert row["bytes_reduction"] == pytest.approx(0.75)
    assert row["wall_change"] == pytest.approx(-0.25)


def test_prediction_accuracy_flags_agreement():
    table = report_module.prediction_accuracy(synthetic_raw())
    assert (table["bytes_error"] == 0).all()
    assert (table["exchange_error"] == 0).all()


def test_thread_policies_are_never_merged():
    """Two policies for the same rank count must stay separate rows."""
    data = synthetic_raw()
    other = data[data["experiment"] == "strong_scaling"].copy()
    other["thread_policy"] = "fixed-total-cores"
    combined = pd.concat([data, other], ignore_index=True)
    summary = report_module.summarise(combined)
    policies = summary[summary["experiment"] == "strong_scaling"]["thread_policy"].unique()
    assert set(policies) == {"one-thread-per-rank", "fixed-total-cores"}


def test_reports_are_written_from_raw_only(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    synthetic_raw().to_csv(raw_dir / "synthetic.csv", index=False)

    written = report_module.write_reports(
        raw=raw_dir, processed=tmp_path / "processed", plots=tmp_path / "plots"
    )
    assert (tmp_path / "processed" / "summary.csv").exists()
    assert "mapping_comparison" in written
    assert (tmp_path / "plots" / "mapping_comparison.png").exists()


def test_missing_raw_data_is_an_error_not_an_empty_plot(tmp_path):
    with pytest.raises(FileNotFoundError, match="no raw measurement files"):
        report_module.load_raw(tmp_path)


def test_markdown_summary_names_the_measurement_host(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    synthetic_raw().to_csv(raw_dir / "synthetic.csv", index=False)
    text = report_module.markdown_summary(raw_dir)
    assert "test-host" in text
    assert "Test CPU" in text


@pytest.mark.slow
def test_readme_results_block_matches_the_committed_measurements():
    """The README quotes measured numbers; they must still follow from the data."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/generate_report.py", "--check", "--skip-reports"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
