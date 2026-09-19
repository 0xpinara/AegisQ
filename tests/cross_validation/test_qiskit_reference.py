"""Phase 3: randomised cross-validation against Qiskit.

Qiskit is used **only** as an external correctness oracle. It is never
imported by the AegisQ runtime; if it disappears the simulator still works and
these tests skip.

The comparison is done with state fidelity rather than elementwise equality
because the two implementations may differ by a global phase — for example
where AegisQ expresses a construction with `rz` and Qiskit with `p`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import Simulator
from tests.conftest import align_global_phase, random_circuit, requires_qiskit, state_fidelity

pytestmark = [pytest.mark.qiskit, requires_qiskit()]

#: A fidelity this close to 1 cannot be reached by a kernel that is subtly
#: wrong on any amplitude of appreciable weight.
FIDELITY_FLOOR = 1.0 - 1e-10


def to_qiskit(circuit: Circuit):
    """Translate an AegisQ circuit into an equivalent Qiskit circuit.

    Both projects use the little-endian convention (qubit `q` is bit `2**q`),
    so no index reversal is needed.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(circuit.num_qubits)
    for gate in circuit:
        method = getattr(qc, gate.opcode)
        method(*gate.params, *gate.qubits)
    return qc


def qiskit_statevector(circuit: Circuit) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(to_qiskit(circuit)).data, dtype=np.complex128)


@pytest.mark.parametrize("backend", ["reference", "cpp"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda c: c.x(0), id="x"),
        pytest.param(lambda c: c.y(1), id="y"),
        pytest.param(lambda c: c.z(2), id="z"),
        pytest.param(lambda c: c.h(0), id="h"),
        pytest.param(lambda c: c.s(1), id="s"),
        pytest.param(lambda c: c.t(2), id="t"),
        pytest.param(lambda c: c.rx(0, 0.7), id="rx"),
        pytest.param(lambda c: c.ry(1, -1.1), id="ry"),
        pytest.param(lambda c: c.rz(2, 2.3), id="rz"),
        pytest.param(lambda c: c.cx(0, 2), id="cx"),
        pytest.param(lambda c: c.cx(2, 0), id="cx-reversed"),
        pytest.param(lambda c: c.cz(1, 2), id="cz"),
        pytest.param(lambda c: c.swap(0, 2), id="swap"),
    ],
)
def test_single_gate_matches_qiskit(backend, build):
    circuit = Circuit(3)
    # Non-trivial prefix so a misplaced operand cannot coincidentally agree.
    circuit.h(0).ry(1, 0.37).rx(2, -0.82).cx(0, 1)
    build(circuit)

    ours = Simulator(backend).run(circuit).statevector
    theirs = qiskit_statevector(circuit)
    assert state_fidelity(ours, theirs) > FIDELITY_FLOOR
    # These gates are defined identically in both projects, so the states
    # should also agree amplitude by amplitude.
    assert np.allclose(ours, theirs, atol=1e-12)


@pytest.mark.parametrize("seed", range(100))
def test_randomised_circuits_match_qiskit(seed):
    num_qubits = 2 + (seed % 5)  # 2..6 qubits
    circuit = random_circuit(num_qubits, depth=8, seed=seed)

    reference = Simulator("reference").run(circuit).statevector
    native = Simulator("cpp").run(circuit).statevector
    theirs = qiskit_statevector(circuit)

    assert state_fidelity(reference, theirs) > FIDELITY_FLOOR, circuit
    assert state_fidelity(native, theirs) > FIDELITY_FLOOR, circuit
    assert np.allclose(align_global_phase(theirs, native), theirs, atol=1e-11)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(25))
def test_deep_wide_circuits_match_qiskit(seed):
    circuit = random_circuit(9, depth=30, seed=1000 + seed)
    ours = Simulator("cpp").run(circuit).statevector
    theirs = qiskit_statevector(circuit)
    assert state_fidelity(ours, theirs) > FIDELITY_FLOOR, circuit


def test_controlled_phase_decomposition_matches_qiskit_cp():
    """AegisQ has no CP gate; it decomposes into rz + cx up to a global phase.

    This is the construction used by the QFT, so it is validated explicitly.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    theta = 0.9
    ours = Circuit(2).h(0).h(1)
    ours.rz(0, theta / 2).rz(1, theta / 2)
    ours.cx(0, 1).rz(1, -theta / 2).cx(0, 1)

    reference = QuantumCircuit(2)
    reference.h(0)
    reference.h(1)
    reference.cp(theta, 0, 1)

    mine = Simulator("cpp").run(ours).statevector
    theirs = np.asarray(Statevector(reference).data)
    # Equal only up to global phase: RZ carries an extra exp(-i theta/2)
    # relative to the phase gate P.
    assert state_fidelity(mine, theirs) > FIDELITY_FLOOR
    assert not np.allclose(mine, theirs, atol=1e-6)


def test_ghz_matches_qiskit_exactly():
    n = 8
    circuit = Circuit(n).h(0)
    for q in range(n - 1):
        circuit.cx(q, q + 1)
    ours = Simulator("cpp").run(circuit).statevector
    theirs = qiskit_statevector(circuit)
    assert np.allclose(ours, theirs, atol=1e-12)
    assert ours[0] == pytest.approx(1 / math.sqrt(2))
