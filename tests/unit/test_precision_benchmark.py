"""Single versus double precision: what it saves and what it costs."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from aegisq.algorithms import build_circuit, random_circuit
from aegisq.benchmark.precision import (
    PRECISION_RAW_FIELDS,
    append_rows,
    compare_precisions,
)
from aegisq.circuit import Circuit
from aegisq.compiler import CommunicationCostModel
from aegisq.compiler.cost_model import default_global_qubits
from aegisq.runtime import Simulator

pytestmark = pytest.mark.native


def test_fp32_halves_predicted_traffic_exactly():
    """Unlike the other levers, this saving needs no search and no analysis."""
    circuit = build_circuit("qft", 14)
    placement = default_global_qubits(14, 8)
    double = CommunicationCostModel(14, 8, "fp64").estimate(circuit, placement)
    single = CommunicationCostModel(14, 8, "fp32").estimate(circuit, placement)
    assert single.bytes_sent * 2 == double.bytes_sent
    assert single.pairwise_exchanges == double.pairwise_exchanges


def test_fp32_halves_the_shard():
    from aegisq.runtime.hardware import memory_estimate

    double = memory_estimate(24, ranks=4, precision="fp64")
    single = memory_estimate(24, ranks=4, precision="fp32")
    assert single["bytes_per_rank"] * 2 == double["bytes_per_rank"]


def test_comparison_records_both_precisions():
    row = compare_precisions(random_circuit(10, depth=6, seed=1), "random", 1)
    assert 0.0 <= row["infidelity"] < 1e-6
    assert row["fidelity"] + row["infidelity"] == pytest.approx(1.0)
    assert row["max_amplitude_error"] >= 0
    assert row["fp64_seconds"] > 0 and row["fp32_seconds"] > 0


def test_error_is_tiny_for_shallow_circuits():
    row = compare_precisions(random_circuit(12, depth=4, seed=2), "random", 2)
    assert row["infidelity"] < 1e-10
    assert row["max_amplitude_error"] < 1e-5


def test_error_grows_with_circuit_size():
    """The honest statement about fp32 is about a depth, not about a precision."""
    shallow = compare_precisions(random_circuit(12, depth=4, seed=3), "random", 3)
    deep = compare_precisions(random_circuit(12, depth=128, seed=3), "random", 3)
    assert deep["gates"] > 10 * shallow["gates"]
    assert deep["infidelity"] > shallow["infidelity"]


def test_error_stays_far_below_sampling_noise():
    """With N shots, sampling noise is ~1/sqrt(N); fp32 error must be smaller."""
    row = compare_precisions(build_circuit("grover", 12), "grover", 0)
    shots = 10**6
    sampling_noise = 1.0 / np.sqrt(shots)
    assert row["infidelity"] < sampling_noise / 1000


def test_norm_is_still_conserved_in_single_precision():
    row = compare_precisions(random_circuit(12, depth=32, seed=4), "random", 4)
    assert row["norm_error"] < 1e-5


def test_fp32_and_fp64_agree_on_the_dominant_outcome():
    circuit = build_circuit("grover", 10).measure_all()
    single = Simulator("cpp", precision="fp32").run(circuit, shots=2000, seed=1)
    double = Simulator("cpp", precision="fp64").run(circuit, shots=2000, seed=1)
    assert single.most_frequent(1)[0][0] == double.most_frequent(1)[0][0]


def test_a_diagonal_only_circuit_loses_almost_nothing():
    """No amplitude mixing means no accumulation."""
    circuit = Circuit(10, name="diagonal")
    for qubit in range(10):
        circuit.h(qubit)
    for _ in range(200):
        for qubit in range(10):
            circuit.rz(qubit, 0.01)
    row = compare_precisions(circuit, "diagonal", 0)
    assert row["infidelity"] < 1e-10


def test_rows_carry_provenance_and_write_to_csv(tmp_path):
    rows = [compare_precisions(random_circuit(8, depth=4, seed=5), "random", 5)]
    path = tmp_path / "precision.csv"
    append_rows(path, rows)
    with path.open(encoding="utf-8") as handle:
        contents = list(csv.reader(handle))
    assert contents[0] == PRECISION_RAW_FIELDS
    assert len(contents) == 2
    assert rows[0]["git_commit"]


def test_infidelity_is_never_negative():
    """`1 - |<a|b>|^2` evaluated directly returns noise near one, including
    negative values, which are not possible fidelities."""
    from aegisq.benchmark.precision import _fidelity

    state = np.zeros(1 << 8, dtype=np.complex128)
    state[3] = 1.0
    nudged = state.copy()
    nudged[3] = 1.0 - 1e-15

    for reference, candidate in ((state, state), (state, nudged), (nudged, state)):
        fidelity, infidelity = _fidelity(reference, candidate)
        assert infidelity >= 0.0
        assert 0.0 <= fidelity <= 1.0


def test_infidelity_matches_the_direct_formula_when_that_is_resolvable():
    """For a difference big enough to survive the subtraction, both agree."""
    from aegisq.benchmark.precision import _fidelity

    a = np.zeros(4, dtype=np.complex128)
    a[0] = 1.0
    b = np.array([np.cos(0.1), np.sin(0.1), 0, 0], dtype=np.complex128)

    _, infidelity = _fidelity(a, b)
    direct = 1 - abs(np.vdot(a, b)) ** 2
    assert infidelity == pytest.approx(direct, rel=1e-9)


def test_global_phase_does_not_count_as_error():
    from aegisq.benchmark.precision import _fidelity

    a = np.array([0.6, 0.8, 0, 0], dtype=np.complex128)
    _, infidelity = _fidelity(a, a * np.exp(1j * 1.234))
    assert infidelity < 1e-25
