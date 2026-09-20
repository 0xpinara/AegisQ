"""How good is the placement search, and why?

The mapper has three strategies. This file pins down when each is used and
what it guarantees:

* **exhaustive** — optimal for the cost model, affordable while `C(n, p)` fits
  the budget;
* **linear** — also optimal, and used when the circuit has no SWAP gates,
  because the objective is then a sum of independent per-qubit terms;
* **greedy + local search** — a heuristic, used only when neither applies.

The separability argument is the interesting one, so it is tested directly
rather than assumed.
"""

from __future__ import annotations

import itertools

import pytest

from aegisq.algorithms import build_circuit, random_circuit
from aegisq.benchmark.placement_quality import compare_strategies
from aegisq.circuit import Circuit
from aegisq.compiler import CommunicationCostModel, StaticCommunicationMapper


def swap_free_circuit(num_qubits: int, depth: int, seed: int) -> Circuit:
    """A random circuit built from everything except SWAP."""
    import random

    rng = random.Random(seed)
    circuit = Circuit(num_qubits, name=f"swapfree{seed}")
    for _ in range(depth):
        for qubit in range(num_qubits):
            opcode = rng.choice(["h", "t", "x", "s", "z", "y"])
            getattr(circuit, opcode)(qubit)
        a, b = rng.sample(range(num_qubits), 2)
        getattr(circuit, rng.choice(["cx", "cz"]))(a, b)
    return circuit


# -- separability -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("world", [4, 8])
def test_objective_is_additive_without_swap_gates(seed, world):
    """bytes(G) == sum of bytes({q}) for every candidate set."""
    circuit = swap_free_circuit(10, 5, seed)
    model = CommunicationCostModel(circuit.num_qubits, world)
    profile = model.compile_circuit(circuit)
    singles = {q: profile.bytes_for((q,)) for q in range(circuit.num_qubits)}

    for candidate in itertools.combinations(range(circuit.num_qubits), model.num_global_qubits):
        assert profile.bytes_for(candidate) == sum(singles[q] for q in candidate)


def test_objective_is_not_additive_with_swap_gates():
    """SWAP costs if *either* operand is global, and an OR is not a sum."""
    circuit = Circuit(6, name="swaps")
    for qubit in range(6):
        circuit.h(qubit)
    circuit.swap(0, 1).swap(0, 2)

    model = CommunicationCostModel(6, 4)
    profile = model.compile_circuit(circuit)
    pair = (0, 1)
    assert profile.bytes_for(pair) < profile.bytes_for((0,)) + profile.bytes_for((1,))


# -- strategy selection -----------------------------------------------------


def test_exhaustive_is_used_when_affordable():
    circuit = swap_free_circuit(10, 4, seed=1)
    result = StaticCommunicationMapper(CommunicationCostModel(10, 4)).optimize(circuit)
    assert result.strategy == "exhaustive"
    assert result.optimal_for_cost_model


def test_linear_strategy_is_used_for_swap_free_circuits_beyond_the_budget():
    circuit = swap_free_circuit(12, 4, seed=2)
    model = CommunicationCostModel(12, 4)
    result = StaticCommunicationMapper(model, candidate_budget=1).optimize(circuit)
    assert result.strategy == "linear (separable objective)"
    assert result.optimal_for_cost_model
    assert "optimal by sorting rather than by search" in result.report()


def test_greedy_is_used_only_when_swaps_make_the_objective_non_separable():
    circuit = random_circuit(12, depth=10, seed=3)
    assert any(gate.opcode == "swap" for gate in circuit)
    result = StaticCommunicationMapper(CommunicationCostModel(12, 4), candidate_budget=1).optimize(
        circuit
    )
    assert result.strategy.startswith("greedy")
    assert not result.optimal_for_cost_model


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("world", [4, 8, 16])
def test_linear_strategy_matches_the_exhaustive_optimum(seed, world):
    circuit = swap_free_circuit(12, 6, seed)
    model = CommunicationCostModel(12, world)
    exhaustive = StaticCommunicationMapper(model).optimize(circuit)
    linear = StaticCommunicationMapper(model, candidate_budget=1).optimize(circuit)
    assert linear.strategy.startswith("linear")
    assert linear.optimized.bytes_sent == exhaustive.optimized.bytes_sent


@pytest.mark.parametrize("family", ["qft", "ising", "grover", "ghz"])
def test_benchmark_families_use_a_strategy_that_matches_the_optimum(family):
    circuit = build_circuit(family, 14, **({"iterations": 2} if family == "grover" else {}))
    model = CommunicationCostModel(circuit.num_qubits, 8)
    exhaustive = StaticCommunicationMapper(model).optimize(circuit)
    fallback = StaticCommunicationMapper(model, candidate_budget=1).optimize(circuit)
    assert fallback.optimized.bytes_sent == exhaustive.optimized.bytes_sent


# -- the measured comparison ------------------------------------------------


def test_comparison_records_both_searches():
    circuit = random_circuit(12, depth=10, seed=5)
    row = compare_strategies(circuit, world_size=4, seed=5)
    assert row["optimal_bytes"] <= row["heuristic_bytes"]
    assert row["gap"] >= 0.0
    assert row["found_optimum"] in (0, 1)
    assert row["candidates"] == 66  # C(12, 2)
    assert row["optimal_seconds"] > 0
    assert row["heuristic_seconds"] > 0


@pytest.mark.parametrize("seed", range(10))
def test_the_heuristic_finds_the_optimum_on_sampled_circuits(seed):
    """Empirical, on circuits that *do* contain swaps, so it could fail."""
    circuit = random_circuit(14, depth=12, seed=seed)
    assert any(gate.opcode == "swap" for gate in circuit)
    row = compare_strategies(circuit, world_size=8, seed=seed)
    assert row["found_optimum"] == 1, f"gap {row['gap']:.3%}"


def test_the_heuristic_is_faster_than_exhaustive_search():
    circuit = random_circuit(18, depth=12, seed=1)
    row = compare_strategies(circuit, world_size=8, seed=1)
    assert row["heuristic_seconds"] < row["optimal_seconds"]
