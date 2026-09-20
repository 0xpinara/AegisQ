"""Property-based tests: invariants that must hold for *every* input.

Hand-written tests check the cases their author thought of. These state the
invariant and let Hypothesis search for a counterexample, which is a different
kind of coverage: it finds the empty circuit, the single-qubit register, the
placement where every qubit is global, and the parameter that happens to be
exactly zero.

Each property is one the implementation genuinely guarantees; a failure here
is a bug in AegisQ, not a mis-stated expectation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aegisq.circuit import Circuit, Gate
from aegisq.circuit.gates import GATE_SPECS
from aegisq.circuit.qasm import parse_qasm, to_qasm
from aegisq.compiler import CommunicationCostModel, optimize_placement
from aegisq.compiler.cost_model import default_global_qubits, mapping_from_global_qubits
from aegisq.provenance.merkle import audit_path, leaf_hash, merkle_root, verify_audit_path
from aegisq.runtime import Simulator
from aegisq.secure.canonical import canonical_bytes, canonical_hash, parse_canonical

#: Only `deadline` is pinned here; the example budget comes from the active
#: Hypothesis profile (registered in tests/conftest.py), so the same
#: properties run as a fast check locally and as a deep search in CI with
#: `--hypothesis-profile=deep`.
SETTINGS = settings(deadline=None)

#: `u` is excluded here and generated separately: it is the internal fused
#: representation, carrying a matrix rather than an angle, so it cannot be
#: built by the same code path as the primitives.
SINGLE_QUBIT_OPS = [op for op, spec in GATE_SPECS.items() if spec.num_qubits == 1 and op != "u"]
TWO_QUBIT_OPS = [op for op, spec in GATE_SPECS.items() if spec.num_qubits == 2]


def _random_unitary(angles: tuple[float, float, float]) -> np.ndarray:
    """A genuine 2x2 unitary built from a product of rotations."""
    from aegisq.circuit.gates import rx_matrix, ry_matrix, rz_matrix

    theta, phi, lam = angles
    return rz_matrix(lam) @ ry_matrix(phi) @ rx_matrix(theta)


@st.composite
def circuits(draw, min_qubits: int = 1, max_qubits: int = 6, max_gates: int = 24):
    """Arbitrary valid circuits over the supported gate set."""
    num_qubits = draw(st.integers(min_value=min_qubits, max_value=max_qubits))
    circuit = Circuit(num_qubits, name="hypothesis")
    gate_count = draw(st.integers(min_value=0, max_value=max_gates))

    for _ in range(gate_count):
        if num_qubits >= 2 and draw(st.booleans()):
            opcode = draw(st.sampled_from(TWO_QUBIT_OPS))
            a, b = draw(
                st.lists(
                    st.integers(min_value=0, max_value=num_qubits - 1),
                    min_size=2,
                    max_size=2,
                    unique=True,
                )
            )
            getattr(circuit, opcode)(a, b)
        elif draw(st.integers(min_value=0, max_value=9)) == 0:
            # Occasionally emit a fused gate directly, so the `u` path is
            # covered by every property rather than only by the fusion tests.
            qubit = draw(st.integers(min_value=0, max_value=num_qubits - 1))
            angles = tuple(
                draw(
                    st.floats(
                        min_value=-math.pi,
                        max_value=math.pi,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                for _ in range(3)
            )
            circuit.u(qubit, _random_unitary(angles))
        else:
            opcode = draw(st.sampled_from(SINGLE_QUBIT_OPS))
            qubit = draw(st.integers(min_value=0, max_value=num_qubits - 1))
            if GATE_SPECS[opcode].num_params:
                angle = draw(
                    st.floats(
                        min_value=-4 * math.pi,
                        max_value=4 * math.pi,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                getattr(circuit, opcode)(qubit, angle)
            else:
                getattr(circuit, opcode)(qubit)
    return circuit


@st.composite
def json_values(draw, depth: int = 2):
    """JSON-compatible values, for the canonical-serialisation properties."""
    scalars = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        st.text(max_size=20),
    )
    if depth <= 0:
        return draw(scalars)
    return draw(
        st.one_of(
            scalars,
            st.lists(json_values(depth - 1), max_size=4),
            st.dictionaries(st.text(max_size=8), json_values(depth - 1), max_size=4),
        )
    )


# -- simulation -------------------------------------------------------------


@SETTINGS
@given(circuit=circuits())
def test_native_and_reference_backends_agree(circuit):
    """The two independent implementations must produce the same state."""
    reference = Simulator("reference").run(circuit).statevector
    native = Simulator("cpp").run(circuit).statevector
    assert np.allclose(reference, native, atol=1e-11), circuit


@SETTINGS
@given(circuit=circuits())
def test_evolution_preserves_the_norm(circuit):
    state = Simulator("cpp").run(circuit).statevector
    assert float(np.vdot(state, state).real) == pytest.approx(1.0, abs=1e-10)


@SETTINGS
@given(circuit=circuits(max_qubits=5, max_gates=16))
def test_inverting_a_circuit_returns_to_the_start(circuit):
    """C followed by C-inverse is the identity, up to a global phase."""
    combined = Circuit(circuit.num_qubits, name="round-trip")
    combined.extend(circuit.gates)
    combined.extend(circuit.inverse().gates)

    state = Simulator("cpp").run(combined).statevector
    expected = np.zeros_like(state)
    expected[0] = 1.0
    assert abs(abs(state[0]) - 1.0) < 1e-9
    assert np.allclose(np.abs(state), np.abs(expected), atol=1e-9)


@SETTINGS
@given(circuit=circuits(min_qubits=1, max_qubits=5), seed=st.integers(0, 2**31 - 1))
def test_sampling_is_reproducible_and_conserves_shots(circuit, seed):
    circuit.measure_all()
    shots = 64
    first = Simulator("cpp").run(circuit, shots=shots, seed=seed).counts
    second = Simulator("cpp").run(circuit, shots=shots, seed=seed).counts
    assert first == second
    assert sum(first.values()) == shots
    assert all(len(key) == circuit.num_qubits for key in first)


@SETTINGS
@given(circuit=circuits(max_qubits=4, max_gates=12))
def test_sampled_outcomes_have_non_zero_probability(circuit):
    """A shot can only land where the state has amplitude."""
    circuit.measure_all()
    result = Simulator("cpp").run(circuit, shots=200, seed=3)
    probabilities = np.abs(Simulator("cpp").run(circuit).statevector) ** 2
    for bitstring in result.counts:
        index = int(bitstring, 2)
        assert probabilities[index] > 1e-12


# -- circuit IR and QASM ----------------------------------------------------


@SETTINGS
@given(circuit=circuits())
def test_dict_round_trip_is_lossless(circuit):
    restored = Circuit.from_dict(circuit.to_dict())
    assert restored.to_dict() == circuit.to_dict()


@SETTINGS
@given(circuit=circuits())
def test_qasm_round_trip_preserves_the_instruction_list(circuit):
    assume(all(gate.opcode != "u" for gate in circuit))  # no QASM form by design
    restored = parse_qasm(to_qasm(circuit))
    assert [str(g) for g in restored] == [str(g) for g in circuit]


@SETTINGS
@given(circuit=circuits(max_qubits=5, max_gates=16))
def test_qasm_round_trip_preserves_the_simulated_state(circuit):
    assume(all(gate.opcode != "u" for gate in circuit))
    restored = parse_qasm(to_qasm(circuit, version="3"))
    assert np.allclose(
        Simulator("cpp").run(restored).statevector,
        Simulator("cpp").run(circuit).statevector,
        atol=1e-12,
    )


@SETTINGS
@given(circuit=circuits())
def test_depth_is_bounded_by_the_gate_count(circuit):
    assert 0 <= circuit.depth() <= len(circuit)


# -- cost model and placement ----------------------------------------------


@st.composite
def model_and_placement(draw):
    """A circuit, a legal rank count and a legal set of global qubits."""
    circuit = draw(circuits(min_qubits=2, max_qubits=8, max_gates=30))
    max_p = circuit.num_qubits - 1
    p = draw(st.integers(min_value=0, max_value=min(3, max_p)))
    world = 1 << p
    chosen = draw(
        st.lists(
            st.integers(min_value=0, max_value=circuit.num_qubits - 1),
            min_size=p,
            max_size=p,
            unique=True,
        )
    )
    return circuit, world, sorted(chosen)


@SETTINGS
@given(case=model_and_placement())
def test_aggregated_scorer_matches_per_gate_accounting(case):
    """The optimiser's fast path must not drift from the readable rules."""
    circuit, world, chosen = case
    model = CommunicationCostModel(circuit.num_qubits, world)
    profile = model.compile_circuit(circuit)
    estimate = model.estimate(circuit, chosen)
    assert profile.bytes_for(chosen) == estimate.bytes_sent
    assert profile.exchanges_for(chosen) == estimate.pairwise_exchanges


@SETTINGS
@given(case=model_and_placement())
def test_predicted_traffic_is_never_negative_and_scales_with_the_shard(case):
    circuit, world, chosen = case
    model = CommunicationCostModel(circuit.num_qubits, world)
    estimate = model.estimate(circuit, chosen)
    assert estimate.bytes_sent >= 0
    assert estimate.pairwise_exchanges >= 0
    assert (estimate.bytes_sent == 0) == (estimate.pairwise_exchanges == 0)
    if world == 1:
        assert estimate.bytes_sent == 0


@SETTINGS
@given(case=model_and_placement())
def test_mapping_helper_round_trips_through_the_cost_model(case):
    circuit, world, chosen = case
    model = CommunicationCostModel(circuit.num_qubits, world)
    mapping = mapping_from_global_qubits(circuit.num_qubits, chosen)
    assert sorted(mapping) == list(range(circuit.num_qubits))
    assert (
        model.estimate_for_mapping(circuit, mapping).bytes_sent
        == model.estimate(circuit, chosen).bytes_sent
    )


@SETTINGS
@given(circuit=circuits(min_qubits=3, max_qubits=8, max_gates=30), p=st.integers(0, 2))
def test_optimizer_never_does_worse_than_the_default_placement(circuit, p):
    assume(circuit.num_qubits > p)
    world = 1 << p
    result = optimize_placement(circuit, world)
    assert result.optimized.bytes_sent <= result.baseline.bytes_sent
    assert 0.0 <= result.reduction <= 1.0
    assert result.baseline.global_qubits == default_global_qubits(circuit.num_qubits, world)
    assert len(result.global_qubits) == p


# -- gate fusion ------------------------------------------------------------


@SETTINGS
@given(circuit=circuits(max_qubits=5, max_gates=20))
def test_fusion_preserves_the_state_exactly(circuit):
    """Exactly, including global phase -- so the comparison is elementwise."""
    from aegisq.compiler import fuse

    assert np.allclose(
        Simulator("cpp").run(fuse(circuit)).statevector,
        Simulator("cpp").run(circuit).statevector,
        atol=1e-12,
    ), circuit


@SETTINGS
@given(case=model_and_placement())
def test_fusion_never_increases_predicted_communication(case):
    from aegisq.compiler import fuse

    circuit, world, chosen = case
    model = CommunicationCostModel(circuit.num_qubits, world)
    assert (
        model.estimate(fuse(circuit), chosen).bytes_sent
        <= model.estimate(circuit, chosen).bytes_sent
    )


@SETTINGS
@given(circuit=circuits(max_qubits=5, max_gates=20))
def test_fusion_reaches_a_fixed_point(circuit):
    """A second pass finds nothing left to merge."""
    from aegisq.compiler import fuse_single_qubit_runs

    once = fuse_single_qubit_runs(circuit).circuit
    twice = fuse_single_qubit_runs(once)
    assert twice.stats.runs_fused == 0
    assert len(twice.circuit) == len(once)


# -- canonical serialisation and Merkle trees -------------------------------


@SETTINGS
@given(value=json_values())
def test_canonical_bytes_round_trip(value):
    assert parse_canonical(canonical_bytes(value)) == value


@SETTINGS
@given(value=json_values())
def test_canonical_hash_depends_only_on_content(value):
    assert canonical_hash(value) == canonical_hash(parse_canonical(canonical_bytes(value)))


@SETTINGS
@given(size=st.integers(min_value=1, max_value=64), data=st.data())
def test_every_merkle_leaf_has_a_verifiable_audit_path(size, data):
    leaves = [leaf_hash(bytes([index % 256])) for index in range(size)]
    root = merkle_root(leaves)
    index = data.draw(st.integers(min_value=0, max_value=size - 1))
    path = audit_path(leaves, index)
    assert verify_audit_path(leaves[index], index, size, path, root)
    assert not verify_audit_path(leaf_hash(b"forged"), index, size, path, root)


@SETTINGS
@given(
    left=st.lists(st.binary(min_size=1, max_size=8), min_size=1, max_size=12),
    right=st.lists(st.binary(min_size=1, max_size=8), min_size=1, max_size=12),
)
def test_distinct_leaf_lists_give_distinct_roots(left, right):
    assume(left != right)
    assert merkle_root([leaf_hash(x) for x in left]) != merkle_root([leaf_hash(x) for x in right])


@SETTINGS
@given(gate=st.sampled_from(sorted(GATE_SPECS)), qubits=st.integers(1, 4))
def test_gate_validation_rejects_out_of_range_operands(gate, qubits):
    spec = GATE_SPECS[gate]
    circuit = Circuit(qubits)
    operands = tuple(range(qubits, qubits + spec.num_qubits))
    params = (0.5,) * spec.num_params
    with pytest.raises(Exception, match="out of range"):
        circuit.append(Gate(gate, operands, params))
