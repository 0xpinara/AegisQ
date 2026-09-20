"""Fused gates on the distributed runtime.

A fused `u` follows the same placement rules as the primitives it replaced:
diagonal ones are free wherever they sit, general ones cost a whole-shard
exchange on a global qubit. These tests check both the arithmetic and the
traffic.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisq.algorithms import build_circuit
from aegisq.circuit import Circuit
from aegisq.compiler import CommunicationCostModel, fuse
from aegisq.compiler.cost_model import default_global_qubits
from aegisq.runtime import Simulator, distributed
from aegisq.runtime.native import to_native_circuit
from tests.conftest import random_circuit

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def geometry(mpi_world):
    p = mpi_world.bit_length() - 1
    num_qubits = p + 5
    lay = distributed.layout(num_qubits)
    return {
        "num_qubits": num_qubits,
        "p": p,
        "local": lay.local_qubits(),
        "global": lay.global_qubits(),
    }


def run(circuit, mapping=None):
    state = distributed.new_distributed_state(circuit.num_qubits, mapping=mapping)
    state.apply_circuit(to_native_circuit(circuit))
    return state


@pytest.mark.parametrize("family", ["qft", "ising", "random", "grover"])
def test_fused_circuits_match_the_reference(geometry, family):
    circuit = build_circuit(family, geometry["num_qubits"])
    fused = fuse(circuit)
    expected = Simulator("reference").run(circuit).statevector
    assert np.allclose(run(fused).gather(), expected, atol=1e-11)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_random_circuits_match_the_reference(geometry, seed):
    circuit = random_circuit(geometry["num_qubits"], depth=8, seed=seed)
    fused = fuse(circuit)
    expected = Simulator("reference").run(circuit).statevector
    assert np.allclose(run(fused).gather(), expected, atol=1e-11)


def test_a_general_fused_gate_on_a_global_qubit_costs_one_exchange(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    target = geometry["global"][0]
    circuit = Circuit(geometry["num_qubits"], name="fused-global")
    for _ in range(6):
        circuit.h(target).t(target)

    unfused_metrics = run(circuit).reduced_metrics()
    fused_metrics = run(fuse(circuit)).reduced_metrics()

    # Twelve gates, six of them non-diagonal, become one fused exchange.
    assert unfused_metrics["pairwise_exchanges"] > fused_metrics["pairwise_exchanges"]
    assert fused_metrics["pairwise_exchanges"] == distributed.world_size()
    assert fused_metrics["bytes_sent"] < unfused_metrics["bytes_sent"]


def test_a_diagonal_fused_gate_on_a_global_qubit_is_free(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no global qubits")
    target = geometry["global"][0]
    circuit = Circuit(geometry["num_qubits"], name="fused-diagonal")
    for qubit in geometry["local"]:
        circuit.h(qubit)
    for _ in range(8):
        circuit.rz(target, 0.3).s(target)

    metrics = run(fuse(circuit)).reduced_metrics()
    assert metrics["bytes_sent"] == 0
    assert metrics["pairwise_exchanges"] == 0


def test_cost_model_predicts_fused_traffic_exactly(geometry):
    circuit = fuse(build_circuit("grover", geometry["num_qubits"]))
    world = distributed.world_size()
    model = CommunicationCostModel(circuit.num_qubits, world)
    estimate = model.estimate(circuit, default_global_qubits(circuit.num_qubits, world))

    metrics = run(circuit).reduced_metrics()
    assert metrics["bytes_sent"] == estimate.bytes_sent
    assert metrics["pairwise_exchanges"] == estimate.pairwise_exchanges


def test_fusion_composes_with_placement(geometry):
    if geometry["p"] == 0:
        pytest.skip("single-rank world has no placement choice")
    from aegisq.compiler import optimize_placement

    world = distributed.world_size()
    circuit = build_circuit("grover", geometry["num_qubits"])
    fused = fuse(circuit)
    placement = optimize_placement(fused, world)

    baseline = run(circuit).reduced_metrics()["bytes_sent"]
    combined = run(fused, list(placement.mapping)).reduced_metrics()["bytes_sent"]
    assert combined <= baseline


def test_fused_measurement_agrees_with_the_unfused_circuit(geometry):
    circuit = random_circuit(geometry["num_qubits"], depth=6, seed=5).measure_all()
    fused = fuse(circuit)
    assert (
        Simulator("mpi").run(fused, shots=512, seed=4).counts
        == Simulator("mpi").run(circuit, shots=512, seed=4).counts
    )
