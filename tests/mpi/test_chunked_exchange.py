"""The chunked-transfer path, forced into range.

MPI element counts are `int`-typed, so the runtime splits an oversized shard
into several calls. At the natural limit that needs a 4 GiB shard, so the code
would never run in a test and would rot. The limit is therefore adjustable,
and these tests drive it down to a handful of elements.

What must stay invariant under chunking:

* the computed state,
* the byte totals and the *logical* exchange count, which is what the cost
  model predicts.

What must change: the count of physical MPI calls, which is a latency-relevant
quantity and is reported separately for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisq import native_core
from aegisq.circuit import Circuit
from aegisq.runtime import Simulator, distributed
from aegisq.runtime.native import to_native_circuit

pytestmark = pytest.mark.mpi


@pytest.fixture
def chunk_limit():
    """Restore the process-wide limit however the test ends.

    The skip belongs here rather than in each test: the fixture is what
    assumes a native core exists. One test in this file took `chunk_limit`
    without also taking `geometry`, and so was the only one that reached a
    `None` core and raised `AttributeError` instead of skipping on a build
    without the extension.
    """
    core = native_core()
    if core is None:
        pytest.skip("native core is not built")
    original = core.max_exchange_elements()
    yield core.set_max_exchange_elements
    core.set_max_exchange_elements(original)


@pytest.fixture
def geometry(mpi_world):
    p = mpi_world.bit_length() - 1
    if p == 0:
        pytest.skip("single-rank world never exchanges")
    num_qubits = p + 5
    lay = distributed.layout(num_qubits)
    return {"num_qubits": num_qubits, "global": lay.global_qubits(), "local": lay.local_qubits()}


def run(circuit, precision="fp64"):
    state = distributed.new_distributed_state(circuit.num_qubits, precision=precision)
    state.apply_circuit(to_native_circuit(circuit))
    return state


def test_chunking_does_not_change_the_result(chunk_limit, geometry):
    circuit = Circuit(geometry["num_qubits"], name="chunked")
    for qubit in range(circuit.num_qubits):
        circuit.h(qubit)
    circuit.cx(geometry["local"][0], geometry["global"][0])

    expected = Simulator("reference").run(circuit).statevector

    chunk_limit(1)  # one amplitude per MPI call
    assert np.allclose(run(circuit).gather(), expected, atol=1e-11)

    chunk_limit(3)  # a size that does not divide the shard evenly
    assert np.allclose(run(circuit).gather(), expected, atol=1e-11)


def test_byte_accounting_is_invariant_under_chunking(chunk_limit, geometry):
    circuit = Circuit(geometry["num_qubits"], name="accounting").h(geometry["global"][0])

    whole = run(circuit).reduced_metrics()
    chunk_limit(4)
    split = run(circuit).reduced_metrics()

    assert split["bytes_sent"] == whole["bytes_sent"]
    assert split["bytes_received"] == whole["bytes_received"]
    assert split["pairwise_exchanges"] == whole["pairwise_exchanges"]


def test_physical_message_count_grows_when_chunked(chunk_limit, geometry):
    circuit = Circuit(geometry["num_qubits"], name="messages").h(geometry["global"][0])

    whole = run(circuit).metrics()
    assert whole["send_calls"] == whole["pairwise_exchanges"]

    shard = 2 ** distributed.layout(geometry["num_qubits"]).num_local_qubits
    chunk_limit(shard // 4)
    split = run(circuit).metrics()

    assert split["pairwise_exchanges"] == whole["pairwise_exchanges"]
    assert split["send_calls"] == 4 * whole["send_calls"]
    assert split["receive_calls"] == split["send_calls"]


def test_per_opcode_message_count_is_reported(chunk_limit, geometry):
    circuit = Circuit(geometry["num_qubits"], name="per-opcode").h(geometry["global"][0])
    shard = 2 ** distributed.layout(geometry["num_qubits"]).num_local_qubits
    chunk_limit(shard // 2)

    metrics = run(circuit).metrics()
    entry = metrics["per_opcode"]["h"]
    assert entry["exchanges"] == 1
    assert entry["messages"] == 2


def test_cost_model_still_matches_when_chunked(chunk_limit, geometry):
    """The model predicts logical exchanges, so chunking must not disturb it."""
    from aegisq.compiler import CommunicationCostModel
    from aegisq.compiler.cost_model import default_global_qubits
    from tests.conftest import random_circuit

    circuit = random_circuit(geometry["num_qubits"], depth=6, seed=4)
    model = CommunicationCostModel(circuit.num_qubits, distributed.world_size())
    estimate = model.estimate(
        circuit, default_global_qubits(circuit.num_qubits, distributed.world_size())
    )

    chunk_limit(2)
    metrics = run(circuit).reduced_metrics()
    assert metrics["bytes_sent"] == estimate.bytes_sent
    assert metrics["pairwise_exchanges"] == estimate.pairwise_exchanges


def test_a_zero_chunk_limit_is_rejected(chunk_limit):
    with pytest.raises(Exception, match="at least one element"):
        chunk_limit(0)
