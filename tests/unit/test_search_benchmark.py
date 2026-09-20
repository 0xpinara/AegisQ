"""Phase 21: Grover query-scaling measurements."""

from __future__ import annotations

import csv

import pytest

from aegisq.algorithms.grover import optimal_iterations
from aegisq.benchmark.search import (
    SEARCH_RAW_FIELDS,
    append_rows,
    measure_grover,
    theoretical_success,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5])
def test_measured_success_tracks_the_theory(bits):
    row = measure_grover(bits, shots=4096, seed=3)
    assert row["measured_success_probability"] == pytest.approx(
        row["theoretical_success_probability"], abs=0.03
    )


def test_grover_uses_far_fewer_queries_than_classical_search():
    row = measure_grover(6, shots=1024, seed=1)
    assert row["grover_iterations"] < row["classical_expected_queries"] / 4
    assert row["classical_worst_case_queries"] == 64


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_query_count_follows_the_square_root(bits):
    import math

    row = measure_grover(bits, shots=64, seed=1)
    assert row["grover_iterations"] == optimal_iterations(bits)
    assert row["grover_iterations"] == math.floor(math.pi / 4 * math.sqrt(2**bits))


def test_theoretical_success_is_a_probability():
    for bits in range(2, 10):
        value = theoretical_success(bits, optimal_iterations(bits))
        assert 0.0 <= value <= 1.0
        assert value > 0.8  # the optimal iteration count should be good


def test_rows_carry_provenance_and_write_to_csv(tmp_path):
    rows = [measure_grover(3, shots=128, seed=2)]
    path = tmp_path / "search.csv"
    append_rows(path, rows)
    with path.open(encoding="utf-8") as handle:
        contents = list(csv.reader(handle))
    assert contents[0] == SEARCH_RAW_FIELDS
    assert len(contents) == 2
    assert rows[0]["git_commit"]
    assert rows[0]["qubits"] >= 3
