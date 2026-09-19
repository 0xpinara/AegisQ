"""Phase 6: non-diagonal single-qubit gates on global qubits.

These are the first operations that actually move amplitudes between ranks.
Each rank exchanges its whole shard with `rank ^ (1 << global_position)` and
then applies one row of the 2x2 matrix; the partner applies the other row.
"""

from __future__ import annotations

import math

import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import distributed
from tests.mpi.conftest import assert_matches_reference, run_distributed

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def geometry(mpi_world):
    p = mpi_world.bit_length() - 1
    num_qubits = p + 4
    lay = distributed.layout(num_qubits)
    return {
        "num_qubits": num_qubits,
        "p": p,
        "local": lay.local_qubits(),
        "global": lay.global_qubits(),
    }


@pytest.mark.parametrize(
    "opcode, params",
    [("x", ()), ("y", ()), ("h", ()), ("rx", (0.83,)), ("ry", (-1.27,))],
)
def test_single_global_gate_matches_reference(geometry, opcode, params):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name=f"global-{opcode}")
    for q in geometry["local"]:
        circuit.h(q)
    for g in geometry["global"]:
        getattr(circuit, opcode)(g, *params)
    assert_matches_reference(circuit)


def test_global_gate_from_the_ground_state(geometry):
    """Starting from |0...0> only rank 0 holds amplitude; H must spread it."""
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="h-on-global")
    circuit.h(geometry["global"][0])
    state = assert_matches_reference(circuit)
    assert state.norm() == pytest.approx(1.0)


def test_hadamard_on_every_qubit(geometry):
    """Uniform superposition: every rank ends up with equal amplitude."""
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="all-h")
    for q in range(n):
        circuit.h(q)
    state = assert_matches_reference(circuit)
    expected_local = 1.0 / state.world_size
    assert state.local_squared_norm() == pytest.approx(expected_local)


def test_global_gate_is_self_inverse(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    n = geometry["num_qubits"]
    g = geometry["global"][0]
    circuit = Circuit(n, name="h-twice")
    for q in geometry["local"]:
        circuit.ry(q, 0.4)
    circuit.h(g).h(g)
    assert_matches_reference(circuit)


def test_rotation_and_its_inverse_cancel_across_ranks(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    n = geometry["num_qubits"]
    g = geometry["global"][-1]
    circuit = Circuit(n, name="rx-inverse")
    for q in geometry["local"]:
        circuit.h(q)
    circuit.rx(g, 1.1).rx(g, -1.1)
    assert_matches_reference(circuit)


def test_every_global_position_is_exercised(geometry):
    """With 4+ ranks there is more than one global position to get right."""
    if geometry["p"] < 2:
        pytest.skip("needs at least two global positions (4 ranks)")
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="all-global-positions")
    for q in geometry["local"]:
        circuit.h(q)
    for g in geometry["global"]:
        circuit.h(g)
        circuit.rz(g, 0.2)
        circuit.ry(g, 0.35)
    assert_matches_reference(circuit)


def test_deep_interleaving_of_local_and_global_gates(geometry):
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="interleaved")
    for layer in range(6):
        for q in range(n):
            circuit.ry(q, 0.13 * (layer + 1) * (q + 1))
        for q in geometry["local"][:-1]:
            circuit.cx(q, q + 1) if (q + 1) in geometry["local"] else circuit.rz(q, 0.1)
        for g in geometry["global"]:
            circuit.h(g)
    assert_matches_reference(circuit)


def test_gates_under_a_reversed_mapping(geometry):
    """With the placement reversed, the *low* logical qubits become global."""
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no placement choice")
    n = geometry["num_qubits"]
    mapping = [n - 1 - q for q in range(n)]
    lay = distributed.layout(n, mapping=mapping)

    circuit = Circuit(n, name="reversed-mapping")
    for q in range(n):
        circuit.h(q)
    for g in lay.global_qubits():
        circuit.rx(g, 0.77)
    assert_matches_reference(circuit, mapping=mapping)


def test_fp32_global_exchange(geometry):
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="fp32-global")
    for q in range(n):
        circuit.h(q)
    for g in geometry["global"]:
        circuit.ry(g, 0.5)
    assert_matches_reference(circuit, precision="fp32", atol=1e-5)


def test_exchange_buffer_is_reused_across_many_gates(geometry):
    """A deep run of global gates must stay correct and not exhaust memory."""
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    n = geometry["num_qubits"]
    g = geometry["global"][0]
    circuit = Circuit(n, name="many-exchanges")
    for q in geometry["local"]:
        circuit.h(q)
    for i in range(40):
        circuit.rx(g, math.pi / 40 if i % 2 == 0 else -math.pi / 40)
    state = run_distributed(circuit)
    assert state.norm() == pytest.approx(1.0, abs=1e-10)
    assert state.gates_applied == len(circuit)
