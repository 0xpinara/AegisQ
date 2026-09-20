"""Distributed correctness when each rank holds almost nothing.

With `n = p + 1` a rank owns exactly two amplitudes, which is where
index arithmetic that works comfortably at 2^17 amplitudes tends to fall
apart: the packed half-shard is a single element, the pair-enumeration loop
runs once, and every "block" degenerates.

The sweep is exhaustive rather than sampled — every gate on every qubit, and
every ordered pair for the two-qubit gates — because at these sizes it is
cheap and the failure modes are individually plausible.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aegisq.circuit import Circuit
from aegisq.runtime import Simulator, distributed
from aegisq.runtime.native import to_native_circuit

pytestmark = pytest.mark.mpi

SINGLE_QUBIT = (
    ("x", ()),
    ("y", ()),
    ("z", ()),
    ("h", ()),
    ("s", ()),
    ("t", ()),
    ("rx", (0.7,)),
    ("ry", (-1.3,)),
    ("rz", (0.4,)),
)


def spread(circuit: Circuit) -> Circuit:
    """Put every qubit in a distinct, non-trivial superposition."""
    for qubit in range(circuit.num_qubits):
        circuit.ry(qubit, 0.3 * (qubit + 1))
    return circuit


def assert_matches_reference(circuit: Circuit, mapping=None) -> None:
    state = distributed.new_distributed_state(circuit.num_qubits, mapping=mapping)
    state.apply_circuit(to_native_circuit(circuit))
    expected = Simulator("reference").run(circuit).statevector
    assert np.allclose(state.gather(), expected, atol=1e-11), circuit.name


@pytest.fixture(params=[1, 2], ids=["local-qubits-1", "local-qubits-2"])
def width(request, mpi_world) -> int:
    """Circuit widths leaving one or two local qubits per rank."""
    p = mpi_world.bit_length() - 1
    n = p + request.param
    if n < 2:
        pytest.skip("needs at least two qubits")
    return n


@pytest.mark.parametrize("opcode, params", SINGLE_QUBIT)
def test_every_single_qubit_gate_on_every_qubit(width, opcode, params):
    for qubit in range(width):
        circuit = spread(Circuit(width, name=f"{opcode}-q{qubit}"))
        getattr(circuit, opcode)(qubit, *params)
        assert_matches_reference(circuit)


@pytest.mark.parametrize("opcode", ["cx", "cz", "swap"])
def test_every_two_qubit_gate_on_every_ordered_pair(width, opcode):
    for a, b in itertools.permutations(range(width), 2):
        circuit = spread(Circuit(width, name=f"{opcode}-{a}-{b}"))
        getattr(circuit, opcode)(a, b)
        assert_matches_reference(circuit)


def test_smallest_shard_still_normalises(width):
    circuit = spread(Circuit(width, name="norm"))
    for qubit in range(width):
        circuit.h(qubit)
    state = distributed.new_distributed_state(width)
    state.apply_circuit(to_native_circuit(circuit))
    assert state.norm() == pytest.approx(1.0, abs=1e-12)
    assert state.local_size == 2**state.layout.num_local_qubits


def test_smallest_shard_under_a_reversed_placement(width, mpi_world):
    if mpi_world == 1:
        pytest.skip("single-rank world has no placement choice")
    mapping = [width - 1 - q for q in range(width)]
    for opcode in ("h", "cx", "swap"):
        circuit = spread(Circuit(width, name=f"reversed-{opcode}"))
        if opcode == "h":
            circuit.h(width - 1)
        else:
            getattr(circuit, opcode)(0, width - 1)
        assert_matches_reference(circuit, mapping=mapping)


def test_measurement_at_the_smallest_shard(width):
    circuit = Circuit(width, name="tiny-measure")
    for qubit in range(width):
        circuit.h(qubit)
    circuit.measure_all()
    distributed_counts = Simulator("mpi").run(circuit, shots=512, seed=9).counts
    single_counts = Simulator("cpp").run(circuit, shots=512, seed=9).counts
    assert distributed_counts == single_counts
