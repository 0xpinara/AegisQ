"""Phase 19: benchmark circuit families and algorithm demonstrations.

The QFT test here compares the *whole matrix* against the DFT. An earlier
version of `qft()` iterated the qubits in the wrong direction and produced a
circuit that was unitary, had the right gate counts, inverted correctly, and
was not the Fourier transform. Only a matrix-level check catches that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegisq.algorithms import CIRCUIT_FAMILIES, build_circuit, ghz, ising_trotter, qft
from aegisq.algorithms.grover import (
    grover,
    multi_controlled_z,
    optimal_iterations,
    toffoli,
)
from aegisq.algorithms.qft import controlled_phase, inverse_qft
from aegisq.algorithms.random_circuit import random_circuit
from aegisq.algorithms.shor import (
    controlled_modular_multiplication,
    factors_from_order,
    inverse_qft_on,
    modular_multiplication_permutation,
    period_from_measurement,
    permutation_transpositions,
    run_shor,
    shor_circuit,
)
from aegisq.circuit import Circuit
from aegisq.runtime import Simulator


def unitary_of(circuit: Circuit) -> np.ndarray:
    """Dense matrix of a circuit, column by column."""
    size = 2**circuit.num_qubits
    columns = []
    for basis in range(size):
        prepared = Circuit(circuit.num_qubits)
        for bit in range(circuit.num_qubits):
            if (basis >> bit) & 1:
                prepared.x(bit)
        prepared.extend(circuit.gates)
        columns.append(Simulator("cpp").run(prepared).statevector)
    return np.array(columns).T


def equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    nonzero = np.abs(b) > atol
    if not nonzero.any():
        return np.allclose(a, b, atol=atol)
    ratio = a[nonzero] / b[nonzero]
    return bool(np.allclose(ratio, ratio.flat[0], atol=atol)) and abs(abs(ratio.flat[0]) - 1) < atol


# -- QFT --------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matrix_equals_the_dft(n):
    size = 2**n
    dft = np.array(
        [[np.exp(2j * np.pi * x * y / size) for x in range(size)] for y in range(size)]
    ) / np.sqrt(size)
    assert equal_up_to_global_phase(unitary_of(qft(n)), dft)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_inverse_qft_undoes_the_qft(n):
    circuit = Circuit(n)
    circuit.extend(qft(n).gates)
    circuit.extend(inverse_qft(n).gates)
    identity = np.eye(2**n)
    assert equal_up_to_global_phase(unitary_of(circuit), identity)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_inverse_qft_on_matches_inverse_qft(n):
    circuit = Circuit(n)
    circuit.extend(qft(n).gates)
    inverse_qft_on(circuit, list(range(n)))
    assert equal_up_to_global_phase(unitary_of(circuit), np.eye(2**n))


def test_qft_of_the_ground_state_is_uniform():
    state = Simulator("cpp").run(qft(4)).statevector
    assert np.allclose(np.abs(state), 0.25, atol=1e-12)


def test_phase_estimation_is_exact_for_a_dyadic_phase():
    """A phase that fits the register must be recovered with probability 1."""
    counting = 4
    for phase, expected in ((0.25, 4), (0.125, 2), (0.75, 12), (0.5, 8)):
        circuit = Circuit(counting)
        for qubit in range(counting):
            circuit.h(qubit)
        for qubit in range(counting):
            circuit.rz(qubit, 2 * math.pi * phase * (2**qubit))
        inverse_qft_on(circuit, list(range(counting)))
        probabilities = np.abs(Simulator("cpp").run(circuit).statevector) ** 2
        assert int(np.argmax(probabilities)) == expected
        assert probabilities.max() == pytest.approx(1.0, abs=1e-9)


def test_controlled_phase_matches_the_ideal_gate():
    theta = math.pi / 3
    circuit = Circuit(2)
    controlled_phase(circuit, 0, 1, theta)
    ideal = np.diag([1, 1, 1, np.exp(1j * theta)])
    assert equal_up_to_global_phase(unitary_of(circuit), ideal)


# -- GHZ and Ising ----------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 9])
def test_ghz_produces_the_cat_state(n):
    state = Simulator("cpp").run(ghz(n)).statevector
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = expected[-1] = 1 / math.sqrt(2)
    assert np.allclose(state, expected, atol=1e-12)


def test_ising_preserves_the_norm_and_has_the_expected_shape():
    circuit = ising_trotter(6, steps=3)
    result = Simulator("cpp").run(circuit)
    assert np.vdot(result.statevector, result.statevector).real == pytest.approx(1.0)
    counts = circuit.gate_counts()
    assert counts["cx"] == 3 * (6 - 1) * 2  # two CX per ZZ interaction
    assert counts["rx"] == 3 * 6


def test_ising_periodic_boundary_adds_one_bond():
    open_chain = ising_trotter(6, steps=1).gate_counts()["cx"]
    ring = ising_trotter(6, steps=1, periodic=True).gate_counts()["cx"]
    assert ring == open_chain + 2


# -- Toffoli, multi-controlled gates, Grover --------------------------------


def test_toffoli_truth_table():
    for a in (0, 1):
        for b in (0, 1):
            for t in (0, 1):
                circuit = Circuit(3)
                if a:
                    circuit.x(0)
                if b:
                    circuit.x(1)
                if t:
                    circuit.x(2)
                toffoli(circuit, 0, 1, 2)
                state = Simulator("cpp").run(circuit).statevector
                expected_t = t ^ (a & b)
                expected = a + 2 * b + 4 * expected_t
                assert abs(abs(state[expected]) - 1) < 1e-10, (a, b, t)


def test_toffoli_is_ccx_up_to_global_phase():
    circuit = Circuit(3)
    toffoli(circuit, 0, 1, 2)
    ccx = np.eye(8, dtype=complex)
    ccx[[3, 7]] = ccx[[7, 3]]
    assert equal_up_to_global_phase(unitary_of(circuit), ccx)


@pytest.mark.parametrize("controls", [1, 2, 3, 4])
def test_multi_controlled_z_phases_only_the_all_ones_state(controls):
    ancillas = max(0, controls - 1)
    total = controls + 1 + ancillas
    control_qubits = list(range(controls))
    target = controls
    ancilla_qubits = list(range(controls + 1, total))

    # The Toffoli ladder contributes a global phase (T-dagger is expressed as
    # an RZ rotation), so amplitudes are compared *relative* to a reference
    # pattern rather than expected to be real.
    amplitudes: dict[int, complex] = {}
    for pattern in range(2 ** (controls + 1)):
        circuit = Circuit(total)
        for bit in range(controls + 1):
            if (pattern >> bit) & 1:
                circuit.x(bit)
        multi_controlled_z(circuit, control_qubits, target, ancilla_qubits)
        state = Simulator("cpp").run(circuit).statevector
        index = int(np.argmax(np.abs(state)))
        assert index == pattern  # basis state unchanged
        assert index >> (controls + 1) == 0  # ancillas returned to |0>
        assert abs(abs(state[index]) - 1) < 1e-10
        amplitudes[pattern] = state[index]

    reference = amplitudes[0]
    all_ones = 2 ** (controls + 1) - 1
    for pattern, amplitude in amplitudes.items():
        expected = -1 if pattern == all_ones else 1
        assert amplitude / reference == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("bits, marked", [(2, 3), (3, 5), (4, 11)])
def test_grover_concentrates_on_the_marked_state(bits, marked):
    circuit = grover(bits, marked=marked)
    result = Simulator("cpp").run(circuit, shots=2000, seed=5)
    top, count = result.most_frequent(1)[0]
    assert int(top, 2) == marked
    assert count / 2000 > 0.8


def test_grover_iteration_count_follows_sqrt_n():
    assert optimal_iterations(4) == 3  # floor(pi/4 * 4)
    assert optimal_iterations(6) == 6  # floor(pi/4 * 8)
    assert optimal_iterations(8) == 12  # floor(pi/4 * 16)


def test_grover_rejects_a_single_search_qubit():
    with pytest.raises(ValueError, match="at least two search qubits"):
        grover(1)


# -- Shor -------------------------------------------------------------------


@pytest.mark.parametrize("modulus, width", [(15, 4), (21, 5)])
def test_modular_multiplication_is_a_permutation(modulus, width):
    for a in range(2, modulus):
        if math.gcd(a, modulus) != 1:
            continue
        permutation = modular_multiplication_permutation(a, modulus, width)
        assert sorted(permutation) == list(range(1 << width))


@pytest.mark.parametrize("modulus, width", [(15, 4), (21, 5)])
def test_transpositions_reproduce_the_permutation(modulus, width):
    for a in range(2, modulus):
        if math.gcd(a, modulus) != 1:
            continue
        permutation = modular_multiplication_permutation(a, modulus, width)
        state = list(range(1 << width))
        applied = list(range(1 << width))
        for x, y in permutation_transpositions(permutation):
            applied[x], applied[y] = applied[y], applied[x]
        # `applied[i]` now holds the label that started at position i's image
        recovered = [0] * len(state)
        for index, label in enumerate(applied):
            recovered[label] = index
        assert recovered == permutation


def test_non_invertible_base_is_rejected():
    with pytest.raises(ValueError, match="not invertible"):
        modular_multiplication_permutation(3, 15, 4)


def test_work_register_must_be_wide_enough():
    with pytest.raises(ValueError, match="cannot hold values"):
        modular_multiplication_permutation(2, 15, 3)


@pytest.mark.parametrize("base", [2, 7])
def test_controlled_modular_multiplication_matches_the_arithmetic(base):
    width, modulus = 4, 15
    ancillas = list(range(1 + width, 1 + width + width - 1))
    work = list(range(1, 1 + width))
    total = 1 + width + len(ancillas)

    for control in (0, 1):
        for value in range(1 << width):
            circuit = Circuit(total)
            if control:
                circuit.x(0)
            for bit in range(width):
                if (value >> bit) & 1:
                    circuit.x(work[bit])
            controlled_modular_multiplication(circuit, 0, work, ancillas, base, modulus)

            state = Simulator("cpp").run(circuit).statevector
            index = int(np.argmax(np.abs(state)))
            assert abs(abs(state[index]) - 1) < 1e-9
            got = (index >> 1) & ((1 << width) - 1)
            expected = (
                ((base * value) % modulus if value < modulus else value) if control else value
            )
            assert got == expected, (base, value, control)
            assert index >> (1 + width) == 0  # ancillas restored


def test_shor_circuit_layout():
    built = shor_circuit(15, 7, counting_qubits=4)
    assert len(built.counting_qubits) == 4
    assert len(built.work_qubits) == 4
    assert built.circuit.measured_qubits == tuple(range(4))


def test_shor_factors_fifteen():
    result = run_shor(15, 7, shots=512, seed=1, counting_qubits=4)
    assert result.succeeded
    assert set(result.factors) == {3, 5}
    assert result.order == 4
    assert "not a claim about factoring" in result.summary()


@pytest.mark.parametrize("base", [2, 4, 7, 11, 13])
def test_shor_factors_fifteen_from_several_bases(base):
    result = run_shor(15, base, shots=512, seed=3, counting_qubits=4)
    assert result.succeeded, result.note
    assert set(result.factors) == {3, 5}


@pytest.mark.slow
def test_shor_factors_twenty_one():
    result = run_shor(21, 2, shots=1024, seed=5, counting_qubits=6)
    assert result.succeeded, result.note
    assert set(result.factors) == {3, 7}


def test_period_recovery_from_a_measurement():
    # 4 counting qubits, r = 4: peaks at 0, 4, 8, 12
    assert period_from_measurement(4, 4, 15) == 4
    assert period_from_measurement(8, 4, 15) == 2  # s/r = 1/2 in lowest terms
    assert period_from_measurement(0, 4, 15) is None


def test_factors_from_order_rejects_useless_orders():
    assert factors_from_order(15, 7, 4) == (3, 5)
    assert factors_from_order(15, 2, 3) is None  # odd order
    assert factors_from_order(15, 14, 2) is None  # a^(r/2) == N-1


def test_even_modulus_is_rejected():
    with pytest.raises(ValueError, match="odd composites"):
        shor_circuit(16, 3)


# -- family registry --------------------------------------------------------


@pytest.mark.parametrize("family", sorted(CIRCUIT_FAMILIES))
def test_every_family_builds_and_runs(family):
    circuit = build_circuit(family, 8)
    result = Simulator("cpp").run(circuit, shots=64, seed=1)
    assert sum(result.counts.values()) == 64
    assert circuit.num_qubits <= 8 + 1


def test_unknown_family_lists_the_available_ones():
    with pytest.raises(ValueError, match="available: ghz, grover, ising, qft, random"):
        build_circuit("nonesuch", 4)


def test_random_circuits_are_reproducible():
    first = random_circuit(6, depth=5, seed=11)
    second = random_circuit(6, depth=5, seed=11)
    assert [str(g) for g in first] == [str(g) for g in second]
    assert [str(g) for g in random_circuit(6, depth=5, seed=12)] != [str(g) for g in first]
