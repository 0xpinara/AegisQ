"""Phase 9: predicted traffic must equal measured traffic.

The cost model drives the optimiser, so a model that is merely plausible would
produce mappings that look good on paper and change nothing on the wire. Every
assertion here compares an analytical prediction against the profiler's
measured byte counts for the same run.
"""

from __future__ import annotations

import pytest

from aegisq.circuit import Circuit
from aegisq.compiler.cost_model import (
    CommunicationCostModel,
    default_global_qubits,
    mapping_from_global_qubits,
)
from aegisq.runtime import distributed
from aegisq.runtime.native import to_native_circuit
from tests.conftest import random_circuit

pytestmark = pytest.mark.mpi


def measured(circuit: Circuit, world: int, mapping=None, precision: str = "fp64"):
    state = distributed.new_distributed_state(
        circuit.num_qubits, precision=precision, mapping=mapping
    )
    state.apply_circuit(to_native_circuit(circuit))
    return state.reduced_metrics()


def predicted(circuit: Circuit, world: int, global_qubits, precision: str = "fp64"):
    model = CommunicationCostModel(circuit.num_qubits, world, precision)
    return model.estimate(circuit, global_qubits)


@pytest.fixture(scope="module")
def width(mpi_world) -> int:
    return (mpi_world.bit_length() - 1) + 5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_prediction_matches_measurement_on_random_circuits(mpi_world, width, seed):
    circuit = random_circuit(width, depth=10, seed=seed)
    globals_ = default_global_qubits(width, mpi_world)

    estimate = predicted(circuit, mpi_world, globals_)
    metrics = measured(circuit, mpi_world)

    assert metrics["bytes_sent"] == estimate.bytes_sent
    assert metrics["bytes_received"] == estimate.bytes_sent
    assert metrics["pairwise_exchanges"] == estimate.pairwise_exchanges


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_prediction_matches_measurement_under_a_custom_mapping(mpi_world, width, seed):
    if mpi_world == 1:
        pytest.skip("single-rank world has no placement choice")
    circuit = random_circuit(width, depth=10, seed=seed)

    # Put the lowest logical qubits on the global positions instead.
    p = mpi_world.bit_length() - 1
    chosen = list(range(p))
    mapping = mapping_from_global_qubits(width, chosen)

    estimate = predicted(circuit, mpi_world, chosen)
    metrics = measured(circuit, mpi_world, mapping=mapping)

    assert metrics["bytes_sent"] == estimate.bytes_sent
    assert metrics["pairwise_exchanges"] == estimate.pairwise_exchanges


def test_zero_cost_mapping_really_sends_nothing(mpi_world, width):
    """A placement the model calls free must move no project-controlled bytes."""
    if mpi_world == 1:
        pytest.skip("single-rank world never communicates")
    p = mpi_world.bit_length() - 1
    chosen = list(range(width - p, width))

    # Only diagonal gates and local operations touch the global qubits.
    circuit = Circuit(width, name="predicted-free")
    for q in range(width - p):
        circuit.h(q).rx(q, 0.3)
    for g in chosen:
        circuit.rz(g, 0.5).z(g)
        circuit.cz(0, g)
        circuit.cx(g, 0)  # global control, local target

    estimate = predicted(circuit, mpi_world, chosen)
    assert estimate.bytes_sent == 0

    metrics = measured(circuit, mpi_world)
    assert metrics["bytes_sent"] == 0
    assert metrics["pairwise_exchanges"] == 0


def test_prediction_matches_for_fp32(mpi_world, width):
    circuit = random_circuit(width, depth=8, seed=21)
    globals_ = default_global_qubits(width, mpi_world)
    estimate = predicted(circuit, mpi_world, globals_, precision="fp32")
    metrics = measured(circuit, mpi_world, precision="fp32")
    assert metrics["bytes_sent"] == estimate.bytes_sent


def test_per_opcode_prediction_matches_measurement(mpi_world, width):
    circuit = random_circuit(width, depth=10, seed=33)
    globals_ = default_global_qubits(width, mpi_world)
    estimate = predicted(circuit, mpi_world, globals_)
    metrics = measured(circuit, mpi_world)

    for opcode, stats in estimate.per_opcode.items():
        entry = metrics["per_opcode"].get(opcode, {})
        assert entry.get("bytes_sent", 0) == stats["bytes_sent"], opcode
        assert entry.get("exchanges", 0) == stats["exchanges"], opcode
        # Gate counts are summed over ranks: every rank executes every gate.
        assert entry.get("gates", 0) == stats["gates"] * mpi_world, opcode


def test_ghz_chain_prediction(mpi_world, width):
    circuit = Circuit(width, name="ghz").h(0)
    for q in range(width - 1):
        circuit.cx(q, q + 1)
    globals_ = default_global_qubits(width, mpi_world)
    estimate = predicted(circuit, mpi_world, globals_)
    metrics = measured(circuit, mpi_world)
    assert metrics["bytes_sent"] == estimate.bytes_sent
