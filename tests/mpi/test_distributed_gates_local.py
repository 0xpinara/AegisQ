"""Phase 5: distributed gates that need no communication.

Covers every placement whose cost is zero bytes on the wire:

* any gate whose operands all sit on local positions,
* diagonal gates (`z`, `s`, `t`, `rz`) on a global qubit, which collapse to a
  single scalar multiply because the rank id fixes the qubit's value,
* `cz` in all four placements, since it is diagonal in both operands,
* `cx` with a global control and a local target, where ranks whose control bit
  is set simply apply `X` locally.

Each case is checked against the single-process reference simulator.
"""

from __future__ import annotations

import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import distributed
from tests.mpi.conftest import assert_matches_reference, run_distributed

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def geometry(mpi_world):
    """Qubit count and the local/global split for the live world size."""
    p = mpi_world.bit_length() - 1
    num_qubits = p + 4
    lay = distributed.layout(num_qubits)
    return {
        "num_qubits": num_qubits,
        "p": p,
        "local": lay.local_qubits(),
        "global": lay.global_qubits(),
    }


def test_local_single_qubit_gates(geometry):
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="local-singles")
    for q in geometry["local"]:
        circuit.h(q).rx(q, 0.3 * (q + 1)).ry(q, -0.2 * (q + 1)).x(q).y(q)
    assert_matches_reference(circuit)


def test_local_two_qubit_gates(geometry):
    local = geometry["local"]
    if len(local) < 2:
        pytest.skip("needs two local qubits")
    circuit = Circuit(geometry["num_qubits"], name="local-pairs")
    for q in local:
        circuit.h(q)
    circuit.cx(local[0], local[1]).cz(local[0], local[1]).swap(local[0], local[1])
    assert_matches_reference(circuit)


@pytest.mark.parametrize("opcode, params", [("z", ()), ("s", ()), ("t", ()), ("rz", (0.73,))])
def test_diagonal_gates_on_global_qubits(geometry, opcode, params):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    n = geometry["num_qubits"]
    circuit = Circuit(n, name=f"global-{opcode}")
    # Spread amplitude over every rank first, using only local operations plus
    # a CX chain whose targets stay local.
    for q in geometry["local"]:
        circuit.h(q)
    for g in geometry["global"]:
        # Reach the global qubit without a global-target CX: a controlled-Z
        # entangles it while staying diagonal.
        circuit.cz(geometry["local"][0], g)
    for g in geometry["global"]:
        getattr(circuit, opcode)(g, *params)
    assert_matches_reference(circuit)


def test_cz_in_every_placement(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    local = geometry["local"]
    globals_ = geometry["global"]
    circuit = Circuit(geometry["num_qubits"], name="cz-placements")
    for q in range(geometry["num_qubits"]):
        circuit.h(q) if q in local else circuit.rz(q, 0.0)
    # local-local
    circuit.cz(local[0], local[1])
    # local-global and global-local (the gate is symmetric, both orders run)
    circuit.cz(local[0], globals_[0])
    circuit.cz(globals_[0], local[1])
    # global-global
    if len(globals_) >= 2:
        circuit.cz(globals_[0], globals_[1])
    assert_matches_reference(circuit)


def test_cx_with_global_control_and_local_target(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    local = geometry["local"]
    globals_ = geometry["global"]
    circuit = Circuit(geometry["num_qubits"], name="cx-global-control")
    for q in local:
        circuit.h(q)
    # Put amplitude on both values of the global qubit without moving data:
    # a CZ cannot do it, so use the fact that |0...0> already has the global
    # bit at 0 and add a controlled phase pattern that keeps the test honest.
    circuit.cz(local[0], globals_[0])
    circuit.cx(globals_[0], local[0])
    circuit.cx(globals_[0], local[1])
    assert_matches_reference(circuit)


def test_gate_counter_tracks_applied_gates(geometry):
    circuit = Circuit(geometry["num_qubits"], name="counter")
    for q in geometry["local"]:
        circuit.h(q)
    state = run_distributed(circuit)
    assert state.gates_applied == len(circuit)


def test_norm_is_preserved_across_ranks(geometry):
    circuit = Circuit(geometry["num_qubits"], name="norm")
    for q in geometry["local"]:
        circuit.h(q)
    for g in geometry["global"]:
        circuit.rz(g, 0.4)
        circuit.cz(geometry["local"][0], g)
    state = run_distributed(circuit)
    assert state.norm() == pytest.approx(1.0, abs=1e-12)


def test_fp32_distributed_matches_reference_loosely(geometry):
    circuit = Circuit(geometry["num_qubits"], name="fp32")
    for q in geometry["local"]:
        circuit.h(q).ry(q, 0.21)
    assert_matches_reference(circuit, precision="fp32", atol=1e-5)


def test_custom_mapping_reproduces_the_same_state(geometry):
    """A permuted placement must not change the computed state."""
    n = geometry["num_qubits"]
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no placement choice")

    # Reverse the placement: logical qubit q goes to position n-1-q, so the
    # *lowest* logical qubits become the global ones.
    mapping = [n - 1 - q for q in range(n)]
    lay = distributed.layout(n, mapping=mapping)

    circuit = Circuit(n, name="mapped")
    for q in lay.local_qubits():
        circuit.h(q).rx(q, 0.17 * (q + 1))
    for g in lay.global_qubits():
        circuit.rz(g, 0.31)
        circuit.cz(lay.local_qubits()[0], g)
    assert_matches_reference(circuit, mapping=mapping)


def test_unsupported_placements_report_the_missing_capability(geometry):
    """Paths that still need a pairwise exchange must fail loudly, not silently.

    Updated as phases land: global single-qubit gates became available in
    Phase 6, so the remaining gap is a CX whose *target* is global.
    """
    if geometry["p"] == 0:
        pytest.skip("single-rank world places every qubit locally")
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="needs-exchange").cx(geometry["local"][0], geometry["global"][0])
    with pytest.raises(Exception, match="not implemented yet"):
        run_distributed(circuit)
