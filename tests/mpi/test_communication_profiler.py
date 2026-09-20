"""Phase 8: measured communication, not modelled communication.

Every byte AegisQ hands to MPI is counted here. These tests pin the byte
counts to closed-form expectations so that a later optimisation cannot quietly
change what "measured traffic" means.

With `S` amplitudes per shard and `w` bytes per amplitude (16 for fp64):

| gate placement | bytes sent per participating rank | participating ranks |
|---|---:|---|
| global `h`/`x`/`ry`  | `S·w`   | all |
| `cx` local control, global target | `S·w/2` | all |
| `cx` global control, global target | `S·w` | half |
| `swap` local/global | `S·w/2` | all |
| `swap` global/global | `S·w` | half |
| anything diagonal, or `cx` with a global control and local target | 0 | none |
"""

from __future__ import annotations

import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import distributed
from aegisq.runtime.native import to_native_circuit

pytestmark = pytest.mark.mpi

BYTES_PER_AMPLITUDE = {"fp64": 16, "fp32": 8}


@pytest.fixture(scope="module")
def geometry(mpi_world):
    p = mpi_world.bit_length() - 1
    num_qubits = p + 5
    lay = distributed.layout(num_qubits)
    return {
        "num_qubits": num_qubits,
        "p": p,
        "world": mpi_world,
        "local": lay.local_qubits(),
        "global": lay.global_qubits(),
        "shard": lay.local_state_size,
    }


def measure(circuit: Circuit, geometry, precision: str = "fp64"):
    """Run a circuit and return (local metrics, world-reduced metrics)."""
    state = distributed.new_distributed_state(circuit.num_qubits, precision=precision, mapping=None)
    state.apply_circuit(to_native_circuit(circuit))
    return state.metrics(), state.reduced_metrics()


def shard_bytes(geometry, precision: str = "fp64") -> int:
    return geometry["shard"] * BYTES_PER_AMPLITUDE[precision]


def test_local_only_circuit_sends_nothing(geometry):
    circuit = Circuit(geometry["num_qubits"], name="local-only")
    for q in geometry["local"]:
        circuit.h(q).rx(q, 0.3)
    local, total = measure(circuit, geometry)
    assert local["bytes_sent"] == 0
    assert total["bytes_sent"] == 0
    assert total["pairwise_exchanges"] == 0
    assert total["communicating_gates"] == 0


def test_diagonal_gates_on_global_qubits_send_nothing(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="global-diagonal")
    for q in geometry["local"]:
        circuit.h(q)
    for g in geometry["global"]:
        circuit.z(g).s(g).t(g).rz(g, 0.4)
        circuit.cz(geometry["local"][0], g)
    local, total = measure(circuit, geometry)
    assert local["bytes_sent"] == 0
    assert total["bytes_sent"] == 0
    assert total["communicating_gates"] == 0


def test_cx_with_global_control_and_local_target_sends_nothing(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="cx-global-control")
    for q in range(geometry["num_qubits"]):
        circuit.h(q)
    circuit.cx(geometry["global"][0], geometry["local"][0])
    _, total = measure(circuit, geometry)
    # The only traffic is from the Hadamards on global qubits.
    expected = shard_bytes(geometry) * geometry["world"] * geometry["p"]
    assert total["bytes_sent"] == expected


def test_global_hadamard_moves_one_whole_state_vector(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="one-global-h")
    circuit.h(geometry["global"][0])
    local, total = measure(circuit, geometry)

    assert local["bytes_sent"] == shard_bytes(geometry)
    assert local["bytes_received"] == shard_bytes(geometry)
    assert local["pairwise_exchanges"] == 1
    # Summed over ranks this is exactly the size of the full state vector.
    assert total["bytes_sent"] == 2 ** geometry["num_qubits"] * 16
    assert total["pairwise_exchanges"] == geometry["world"]


def test_cx_with_global_target_sends_half_a_shard(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="cx-global-target")
    circuit.cx(geometry["local"][0], geometry["global"][0])
    local, total = measure(circuit, geometry)

    assert local["bytes_sent"] == shard_bytes(geometry) // 2
    assert total["bytes_sent"] == shard_bytes(geometry) // 2 * geometry["world"]
    assert total["communicating_gates"] == geometry["world"]


def test_cx_between_two_global_qubits_involves_half_the_ranks(geometry):
    if geometry["p"] < 2:
        pytest.skip("needs two global qubits (4 ranks)")
    circuit = Circuit(geometry["num_qubits"], name="cx-global-global")
    circuit.cx(geometry["global"][0], geometry["global"][1])
    _, total = measure(circuit, geometry)

    # Only ranks whose control bit is set exchange, and they send a whole shard.
    assert total["bytes_sent"] == shard_bytes(geometry) * (geometry["world"] // 2)
    assert total["pairwise_exchanges"] == geometry["world"] // 2


def test_swap_local_global_sends_half_a_shard(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="swap-local-global")
    circuit.swap(geometry["local"][0], geometry["global"][0])
    local, total = measure(circuit, geometry)
    assert local["bytes_sent"] == shard_bytes(geometry) // 2
    assert total["bytes_sent"] == shard_bytes(geometry) // 2 * geometry["world"]


def test_swap_global_global_involves_half_the_ranks(geometry):
    if geometry["p"] < 2:
        pytest.skip("needs two global qubits (4 ranks)")
    circuit = Circuit(geometry["num_qubits"], name="swap-global-global")
    circuit.swap(geometry["global"][0], geometry["global"][1])
    _, total = measure(circuit, geometry)
    assert total["bytes_sent"] == shard_bytes(geometry) * (geometry["world"] // 2)


def test_fp32_shards_move_half_the_bytes(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="fp32-traffic")
    circuit.h(geometry["global"][0])
    local64, _ = measure(circuit, geometry, precision="fp64")
    local32, _ = measure(circuit, geometry, precision="fp32")
    assert local32["bytes_sent"] * 2 == local64["bytes_sent"]


def test_per_opcode_breakdown_attributes_traffic_correctly(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    circuit = Circuit(geometry["num_qubits"], name="mixed")
    for q in geometry["local"]:
        circuit.h(q)
    circuit.rz(geometry["global"][0], 0.25)  # diagonal: free
    circuit.h(geometry["global"][0])  # full shard
    circuit.cx(geometry["local"][0], geometry["global"][0])  # half shard

    local, _ = measure(circuit, geometry)
    per_opcode = local["per_opcode"]

    assert per_opcode["rz"]["bytes_sent"] == 0
    assert per_opcode["h"]["bytes_sent"] == shard_bytes(geometry)
    assert per_opcode["cx"]["bytes_sent"] == shard_bytes(geometry) // 2
    assert per_opcode["h"]["gates"] == len(geometry["local"]) + 1


def test_timings_are_consistent(geometry):
    circuit = Circuit(geometry["num_qubits"], name="timing")
    for q in range(geometry["num_qubits"]):
        circuit.h(q)
    local, total = measure(circuit, geometry)

    assert local["total_seconds"] > 0.0
    assert local["communication_seconds"] >= 0.0
    assert local["compute_seconds"] >= 0.0
    # Compute time is derived as total minus communication, so they add up.
    assert local["compute_seconds"] + local["communication_seconds"] == pytest.approx(
        local["total_seconds"], rel=1e-9
    )
    assert total["total_seconds"] >= local["total_seconds"] - 1e-12


def test_norm_is_counted_as_a_collective(geometry):
    state = distributed.new_distributed_state(geometry["num_qubits"])
    assert state.metrics()["allreduce_calls"] == 0
    state.norm()
    expected = 1 if geometry["world"] > 1 else 0
    assert state.metrics()["allreduce_calls"] == expected


def test_reset_metrics_clears_counters(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    state = distributed.new_distributed_state(geometry["num_qubits"])
    circuit = Circuit(geometry["num_qubits"]).h(geometry["global"][0])
    state.apply_circuit(to_native_circuit(circuit))
    assert state.metrics()["bytes_sent"] > 0
    state.reset_metrics()
    assert state.metrics()["bytes_sent"] == 0
    assert state.metrics()["gates_applied"] == 0
