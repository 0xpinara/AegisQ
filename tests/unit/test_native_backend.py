"""Phase 2: the C++ engine must agree with the NumPy reference exactly.

The reference implementation is authoritative. Any disagreement here means the
native kernels are wrong, not the other way round.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import Simulator
from aegisq.runtime.native import counts_to_bitstrings, new_state
from tests.conftest import random_circuit

pytestmark = pytest.mark.native


def both_backends(circuit: Circuit, precision: str = "fp64"):
    reference = Simulator("reference", precision=precision).run(circuit).statevector
    native = Simulator("cpp", precision=precision).run(circuit).statevector
    return reference, native


@pytest.mark.parametrize("opcode", ["x", "y", "z", "h", "s", "t"])
@pytest.mark.parametrize("qubit", [0, 1, 3])
def test_single_qubit_gates_match_reference(native_core, opcode, qubit, tol):
    circuit = Circuit(4)
    # Prepare an asymmetric state so a wrong qubit index cannot pass by luck.
    circuit.h(0).ry(1, 0.3).rx(2, -0.7).h(3).cx(0, 2)
    getattr(circuit, opcode)(qubit)
    reference, native = both_backends(circuit)
    assert np.allclose(reference, native, atol=tol)


@pytest.mark.parametrize("opcode", ["rx", "ry", "rz"])
@pytest.mark.parametrize("theta", [0.0, 0.4, -1.9, math.pi / 3])
def test_rotations_match_reference(native_core, opcode, theta, tol):
    circuit = Circuit(4)
    circuit.h(0).cx(0, 1).ry(2, 0.2)
    getattr(circuit, opcode)(2, theta)
    reference, native = both_backends(circuit)
    assert np.allclose(reference, native, atol=tol)


@pytest.mark.parametrize("opcode", ["cx", "cz", "swap"])
@pytest.mark.parametrize("pair", [(0, 1), (1, 0), (0, 4), (4, 0), (2, 3)])
def test_two_qubit_gates_match_reference(native_core, opcode, pair, tol):
    circuit = Circuit(5)
    for q in range(5):
        circuit.ry(q, 0.11 * (q + 1))
    getattr(circuit, opcode)(*pair)
    reference, native = both_backends(circuit)
    assert np.allclose(reference, native, atol=tol)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_circuits_match_reference(native_core, num_qubits, seed, tol):
    circuit = random_circuit(num_qubits, depth=12, seed=seed)
    reference, native = both_backends(circuit)
    assert np.allclose(reference, native, atol=tol), circuit


def test_wide_circuit_exercises_the_threaded_path(native_core, tol):
    # 14 qubits = 16384 amplitudes, above the OpenMP threshold of 4096 pairs.
    circuit = random_circuit(14, depth=6, seed=17)
    reference, native = both_backends(circuit)
    assert np.allclose(reference, native, atol=tol)


def test_norm_is_preserved(native_core):
    state = new_state(12, "fp64")
    for q in range(12):
        state.apply_h(q)
    for q in range(11):
        state.apply_cnot(q, q + 1)
    assert state.norm() == pytest.approx(1.0, abs=1e-12)


def test_fp32_backend_tracks_fp64(native_core):
    circuit = random_circuit(6, depth=8, seed=23)
    single = Simulator("cpp", precision="fp32").run(circuit).statevector
    double = Simulator("cpp", precision="fp64").run(circuit).statevector
    assert single.dtype == np.complex64
    assert np.allclose(single, double, atol=1e-5)


def test_counts_agree_with_probabilities(native_core):
    circuit = Circuit(3).h(0).cx(0, 1).measure_all()
    result = Simulator("cpp").run(circuit, shots=20000, seed=11)
    assert set(result.counts) == {"000", "011"}
    assert sum(result.counts.values()) == 20000
    # 20k shots of a fair coin: 5 sigma is ~350 shots.
    assert abs(result.counts["000"] - 10000) < 500


def test_sampling_is_reproducible(native_core):
    circuit = random_circuit(6, depth=6, seed=5).measure_all()
    first = Simulator("cpp").run(circuit, shots=2000, seed=99).counts
    second = Simulator("cpp").run(circuit, shots=2000, seed=99).counts
    assert first == second


def test_partial_measurement_marginalises(native_core):
    circuit = Circuit(3).h(0).x(2).measure(2)
    counts = Simulator("cpp").run(circuit, shots=100, seed=4).counts
    assert counts == {"1": 100}


def test_counts_to_bitstrings_orders_high_qubit_first():
    # index 5 = 0b101 -> q0=1, q1=0, q2=1
    assert counts_to_bitstrings({5: 3}, [0, 1, 2]) == {"101": 3}
    assert counts_to_bitstrings({5: 3}, [0, 2]) == {"11": 3}
    assert counts_to_bitstrings({5: 3}, [1]) == {"0": 3}


def test_native_metrics_are_reported(native_core):
    circuit = random_circuit(8, depth=5, seed=2)
    result = Simulator("cpp").run(circuit, shots=10, seed=1)
    assert result.metrics["gates"] == len(circuit)
    assert result.metrics["compute_seconds"] >= 0.0
    assert result.metrics["threads"] >= 1


def test_invalid_operands_are_rejected(native_core):
    state = new_state(3)
    with pytest.raises(Exception, match="out of range"):
        state.apply_h(9)
    with pytest.raises(Exception, match="distinct"):
        state.apply_cnot(1, 1)


def test_unknown_precision_is_rejected(native_core):
    with pytest.raises(ValueError, match="precision"):
        new_state(2, "fp16")


def test_collective_methods_say_that_they_are_collective():
    """Calling one of these on a single rank hangs the whole job.

    They are MPI_Allreduce / MPI_Allgather underneath, so a rank that
    skips the call never enters the reduction the others are waiting in,
    and the job stops with no error and no output. I hit this writing a
    probe with `if rank() == 0: print(state.norm())` and spent a while
    assuming the simulator had deadlocked. The docstring is the only
    warning the caller gets, so it has to be there.
    """
    from aegisq import native_core

    core = native_core()
    if core is None:
        pytest.skip("native core is not built")

    state_class = core.DistributedStateVectorF64
    for name in ("norm", "gather", "measure_all", "reduced_metrics"):
        doc = getattr(state_class, name).__doc__ or ""
        assert "COLLECTIVE" in doc, f"{name} does not warn that it is collective"

    # And the one that deliberately is not, so the distinction stays useful.
    assert "Not collective" in (state_class.local_squared_norm.__doc__ or "")
