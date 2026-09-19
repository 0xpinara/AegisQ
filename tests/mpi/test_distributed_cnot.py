"""Phase 7: the four placement cases of a distributed CX, plus global SWAP.

| case | control | target  | communication |
|------|---------|---------|---------------|
| A    | local   | local   | none          |
| B    | global  | local   | none          |
| C    | local   | global  | half a shard  |
| D    | global  | global  | a full shard, for half of the ranks |

Each case is forced explicitly and compared against the single-process
reference, because a CX that is merely "usually right" would still produce a
plausible-looking benchmark.
"""

from __future__ import annotations

import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import distributed
from tests.conftest import random_circuit
from tests.mpi.conftest import assert_matches_reference

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


def spread(circuit: Circuit) -> Circuit:
    """Put every qubit in superposition so no case can pass on a sparse state."""
    for q in range(circuit.num_qubits):
        circuit.h(q)
        circuit.rz(q, 0.11 * (q + 1))
    return circuit


def test_case_a_control_local_target_local(geometry):
    local = geometry["local"]
    if len(local) < 2:
        pytest.skip("needs two local qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="cx-local-local"))
    circuit.cx(local[0], local[1])
    assert_matches_reference(circuit)


def test_case_b_control_global_target_local(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="cx-global-local"))
    circuit.cx(geometry["global"][0], geometry["local"][0])
    assert_matches_reference(circuit)


def test_case_c_control_local_target_global(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="cx-local-global"))
    circuit.cx(geometry["local"][0], geometry["global"][0])
    assert_matches_reference(circuit)


def test_case_d_control_global_target_global(geometry):
    if geometry["p"] < 2:
        pytest.skip("needs two global qubits (4 ranks)")
    circuit = spread(Circuit(geometry["num_qubits"], name="cx-global-global"))
    circuit.cx(geometry["global"][0], geometry["global"][1])
    assert_matches_reference(circuit)


def test_case_c_on_every_local_control_position(geometry):
    """The packed half-shard exchange is indexed by the control's position."""
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    for control in geometry["local"]:
        circuit = spread(Circuit(geometry["num_qubits"], name=f"cx-c{control}"))
        circuit.cx(control, geometry["global"][0])
        assert_matches_reference(circuit)


def test_cnot_on_the_ground_state_is_a_no_op(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="cx-ground")
    circuit.cx(geometry["local"][0], geometry["global"][0])
    assert_matches_reference(circuit)


def test_cnot_is_self_inverse_across_ranks(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="cx-twice"))
    circuit.cx(geometry["local"][0], geometry["global"][0])
    circuit.cx(geometry["local"][0], geometry["global"][0])
    assert_matches_reference(circuit)


def test_ghz_chain_crosses_every_rank_boundary(geometry):
    """A GHZ chain over all qubits necessarily uses global-target CX gates."""
    n = geometry["num_qubits"]
    circuit = Circuit(n, name="ghz-chain").h(0)
    for q in range(n - 1):
        circuit.cx(q, q + 1)
    state = assert_matches_reference(circuit)
    assert state.norm() == pytest.approx(1.0)


def test_swap_local_global(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="swap-local-global"))
    circuit.swap(geometry["local"][0], geometry["global"][0])
    assert_matches_reference(circuit)


def test_swap_global_local_operand_order(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="swap-global-local"))
    circuit.swap(geometry["global"][0], geometry["local"][1])
    assert_matches_reference(circuit)


def test_swap_global_global(geometry):
    if geometry["p"] < 2:
        pytest.skip("needs two global qubits (4 ranks)")
    circuit = spread(Circuit(geometry["num_qubits"], name="swap-global-global"))
    circuit.swap(geometry["global"][0], geometry["global"][1])
    assert_matches_reference(circuit)


def test_swap_is_self_inverse(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = spread(Circuit(geometry["num_qubits"], name="swap-twice"))
    circuit.swap(geometry["local"][0], geometry["global"][0])
    circuit.swap(geometry["local"][0], geometry["global"][0])
    assert_matches_reference(circuit)


def test_swap_moves_a_basis_state(geometry):
    """|...1_local...> must end up as |...1_global...> after the swap."""
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    local = geometry["local"][0]
    global_ = geometry["global"][0]
    circuit = Circuit(geometry["num_qubits"], name="swap-basis").x(local).swap(local, global_)
    state = assert_matches_reference(circuit)
    gathered = state.gather()
    expected_index = 1 << global_
    assert abs(gathered[expected_index]) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_random_circuits_match_the_reference(geometry, seed):
    """Full mixed workload: every opcode, every placement, random order."""
    circuit = random_circuit(geometry["num_qubits"], depth=10, seed=seed)
    assert_matches_reference(circuit)


@pytest.mark.parametrize("seed", [11, 12])
def test_random_circuits_under_a_reversed_mapping(geometry, seed):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no placement choice")
    n = geometry["num_qubits"]
    mapping = [n - 1 - q for q in range(n)]
    circuit = random_circuit(n, depth=10, seed=seed)
    assert_matches_reference(circuit, mapping=mapping)


@pytest.mark.parametrize("seed", [21])
def test_random_circuits_in_fp32(geometry, seed):
    circuit = random_circuit(geometry["num_qubits"], depth=8, seed=seed)
    assert_matches_reference(circuit, precision="fp32", atol=1e-5)
