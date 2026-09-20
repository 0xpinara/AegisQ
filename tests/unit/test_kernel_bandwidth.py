"""Local kernel bandwidth measurement.

The numbers are timings and cannot be asserted, but the accounting can: the
byte model per kernel, the reference shapes, and the schema of a raw row.
"""

from __future__ import annotations

import csv

import pytest

from aegisq import native_core
from aegisq.benchmark.kernels import (
    _TOUCHED_FRACTION,
    KERNEL_RAW_FIELDS,
    append_rows,
    measure_kernel,
)

pytestmark = pytest.mark.native


@pytest.fixture(autouse=True)
def restore_threads():
    core = native_core()
    original = core.max_threads()
    yield
    core.set_num_threads(original)


def test_reference_shapes_are_available():
    core = native_core()
    for name in ("stream_triad", "stream_copy", "stream_scale_in_place"):
        sample = getattr(core, name)(1 << 16, 2)
        assert sample["seconds"] > 0
        assert sample["bytes"] > 0
        assert sample["gb_per_second"] > 0


def test_in_place_reference_counts_two_streams_over_one_array():
    core = native_core()
    elements = 1 << 16
    sample = core.stream_scale_in_place(elements, 2)
    assert sample["bytes"] == elements * 16 * 2


def test_triad_counts_three_streams():
    core = native_core()
    elements = 1 << 16
    assert core.stream_triad(elements, 2)["bytes"] == elements * 16 * 3


@pytest.mark.parametrize("kernel", sorted(_TOUCHED_FRACTION))
def test_byte_model_matches_the_fraction_the_kernel_touches(kernel):
    qubits = 12
    row = measure_kernel(kernel, qubits, target_qubit=3, threads=1, repeats=2)
    expected = int((1 << qubits) * _TOUCHED_FRACTION[kernel] * 16 * 2)
    assert row["bytes_per_gate"] == expected


def test_full_sweep_kernels_move_more_than_partial_ones():
    """A general single-qubit gate touches everything; CZ touches a quarter."""
    full = measure_kernel("h", 12, 3, threads=1, repeats=2)
    partial = measure_kernel("cz", 12, 3, threads=1, repeats=2)
    assert full["bytes_per_gate"] == 4 * partial["bytes_per_gate"]


def test_row_reports_achieved_bandwidth_against_the_reference():
    row = measure_kernel("h", 14, 5, threads=1, repeats=3)
    assert row["gb_per_second"] > 0
    assert row["inplace_gb_per_second"] > 0
    assert row["fraction_of_inplace"] == pytest.approx(
        row["gb_per_second"] / row["inplace_gb_per_second"]
    )
    assert row["seconds_per_gate"] > 0


def test_thread_count_is_recorded_as_actually_applied():
    row = measure_kernel("h", 12, 3, threads=2, repeats=2)
    assert row["threads"] == native_core().max_threads()


def test_rows_carry_provenance_and_write_to_csv(tmp_path):
    rows = [measure_kernel("rz", 12, 2, threads=1, repeats=2)]
    path = tmp_path / "kernels.csv"
    append_rows(path, rows)
    with path.open(encoding="utf-8") as handle:
        contents = list(csv.reader(handle))
    assert contents[0] == KERNEL_RAW_FIELDS
    assert len(contents) == 2
    assert rows[0]["cpu_model"]
    assert rows[0]["git_commit"]


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="unknown kernel"):
        measure_kernel("nonesuch", 10, 1, threads=1, repeats=1)


def test_position_table_reports_the_spread():
    import pandas as pd

    from aegisq.benchmark.report import kernel_position_table

    rows = [measure_kernel("h", 12, target, threads=1, repeats=2) for target in (1, 6, 11)]
    table = kernel_position_table(pd.DataFrame(rows))
    assert len(table) == 1
    assert 0.0 <= table["spread"].iloc[0] < 1.0
