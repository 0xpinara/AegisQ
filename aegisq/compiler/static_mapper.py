"""Choose which logical qubits live on the rank-selecting (global) positions.

The runtime's default placement — the identity mapping — makes the
highest-numbered qubits global, which is an accident of indexing rather than a
decision. This module makes the decision deliberately: it searches the
placement space for the set of `p = log2(P)` global qubits that minimises
predicted MPI traffic for a specific circuit.

Search strategy
---------------
Predicted byte volume depends only on the *set* of global qubits (see
`docs/optimizer.md`), so the space is `C(n, p)` subsets — 276 for 24 qubits on
4 ranks, 2024 on 8 ranks. When that is small enough the search is exhaustive
and therefore optimal *with respect to the cost model*. Beyond a configurable
budget it falls back to a greedy start plus pairwise local search, which is
not optimal and says so in its result.

The result is applied, not just reported: `MappingResult.mapping` is the
permutation handed to the distributed runtime.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from aegisq.circuit.circuit import Circuit
from aegisq.compiler.cost_model import (
    CircuitCostProfile,
    CommunicationCostModel,
    CommunicationEstimate,
    default_global_qubits,
    mapping_from_global_qubits,
)

#: Above this many candidate subsets the exhaustive search is abandoned.
#: 200k candidates is roughly a second of scoring with the aggregated profile.
DEFAULT_CANDIDATE_BUDGET = 200_000


@dataclass(frozen=True)
class MappingResult:
    """Outcome of a placement search."""

    global_qubits: tuple[int, ...]
    mapping: tuple[int, ...]
    baseline: CommunicationEstimate
    optimized: CommunicationEstimate
    strategy: str
    candidates_evaluated: int
    search_seconds: float
    optimal_for_cost_model: bool

    @property
    def bytes_saved(self) -> int:
        return self.baseline.bytes_sent - self.optimized.bytes_sent

    @property
    def reduction(self) -> float:
        """Fraction of predicted bytes removed, in [0, 1]."""
        if self.baseline.bytes_sent == 0:
            return 0.0
        return self.bytes_saved / self.baseline.bytes_sent

    def improved(self) -> bool:
        return self.optimized.bytes_sent < self.baseline.bytes_sent or (
            self.optimized.bytes_sent == self.baseline.bytes_sent
            and self.optimized.pairwise_exchanges < self.baseline.pairwise_exchanges
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_qubits": list(self.global_qubits),
            "mapping": list(self.mapping),
            "baseline": self.baseline.as_dict(),
            "optimized": self.optimized.as_dict(),
            "strategy": self.strategy,
            "candidates_evaluated": self.candidates_evaluated,
            "search_seconds": self.search_seconds,
            "optimal_for_cost_model": self.optimal_for_cost_model,
            "predicted_reduction": self.reduction,
        }

    def report(self) -> str:
        """Human-readable before/after comparison."""
        lines = [
            "Baseline mapping (default placement):",
            f"  global qubits:            {list(self.baseline.global_qubits)}",
            f"  predicted MPI traffic:    {_format_bytes(self.baseline.bytes_sent)}",
            f"  predicted exchanges:      {self.baseline.pairwise_exchanges}",
            f"  communicating gates:      {self.baseline.communicating_gates}",
            "",
            "Optimized mapping (communication-aware):",
            f"  global qubits:            {list(self.optimized.global_qubits)}",
            f"  predicted MPI traffic:    {_format_bytes(self.optimized.bytes_sent)}",
            f"  predicted exchanges:      {self.optimized.pairwise_exchanges}",
            f"  communicating gates:      {self.optimized.communicating_gates}",
            "",
            f"Predicted reduction:        {self.reduction * 100:.1f}%",
            f"Search:                     {self.strategy}, "
            f"{self.candidates_evaluated} candidate(s) in {self.search_seconds * 1000:.1f} ms",
        ]
        if not self.optimal_for_cost_model:
            lines.append("Note: heuristic search; the result is not guaranteed optimal.")
        lines.append(
            "These are cost-model predictions. Measured traffic is reported by "
            "`aegisq benchmark` runs."
        )
        return "\n".join(lines)


def _format_bytes(value: int) -> str:
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= scale:
            return f"{value / scale:.2f} {unit} ({value} B)"
    return f"{value} B"


class StaticCommunicationMapper:
    """Searches for a communication-minimising set of global qubits."""

    def __init__(
        self,
        model: CommunicationCostModel,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> None:
        self.model = model
        self.candidate_budget = candidate_budget

    # -- public API ---------------------------------------------------------

    def optimize(
        self,
        circuit: Circuit,
        baseline_global_qubits: Sequence[int] | None = None,
    ) -> MappingResult:
        """Find the best placement this mapper can for `circuit`."""
        if circuit.num_qubits != self.model.num_qubits:
            raise ValueError(
                f"circuit has {circuit.num_qubits} qubits, cost model expects "
                f"{self.model.num_qubits}"
            )

        p = self.model.num_global_qubits
        baseline_set = tuple(
            baseline_global_qubits
            if baseline_global_qubits is not None
            else default_global_qubits(self.model.num_qubits, self.model.world_size)
        )
        baseline = self.model.estimate(circuit, baseline_set)

        if p == 0:
            # A single rank never communicates; nothing to choose.
            return MappingResult(
                global_qubits=(),
                mapping=tuple(range(self.model.num_qubits)),
                baseline=baseline,
                optimized=baseline,
                strategy="trivial (single rank)",
                candidates_evaluated=0,
                search_seconds=0.0,
                optimal_for_cost_model=True,
            )

        profile = self.model.compile_circuit(circuit)
        started = time.perf_counter()
        total_candidates = math.comb(self.model.num_qubits, p)

        if total_candidates <= self.candidate_budget:
            best, evaluated = self._exhaustive(profile, p)
            strategy = "exhaustive"
            optimal = True
        else:
            best, evaluated = self._greedy_with_local_search(profile, p)
            strategy = "greedy + pairwise local search"
            optimal = False

        elapsed = time.perf_counter() - started
        optimized = self.model.estimate(circuit, best)

        return MappingResult(
            global_qubits=tuple(sorted(best)),
            mapping=tuple(mapping_from_global_qubits(self.model.num_qubits, best)),
            baseline=baseline,
            optimized=optimized,
            strategy=strategy,
            candidates_evaluated=evaluated,
            search_seconds=elapsed,
            optimal_for_cost_model=optimal,
        )

    # -- search strategies --------------------------------------------------

    def _score(self, profile: CircuitCostProfile, candidate: Iterable[int]) -> tuple[int, int]:
        """Lexicographic objective: bytes first, then message count.

        Bytes are the primary target because bandwidth is what placement
        actually controls. Message count breaks ties — two placements can move
        identical volume while one does it in half as many, larger transfers.
        """
        candidate = tuple(candidate)
        return (profile.bytes_for(candidate), profile.exchanges_for(candidate))

    def _exhaustive(self, profile: CircuitCostProfile, p: int) -> tuple[tuple[int, ...], int]:
        best: tuple[int, ...] | None = None
        best_score: tuple[int, int] | None = None
        evaluated = 0

        for candidate in itertools.combinations(range(self.model.num_qubits), p):
            score = self._score(profile, candidate)
            evaluated += 1
            if best_score is None or score < best_score:
                best, best_score = candidate, score
        assert best is not None
        return best, evaluated

    def _greedy_with_local_search(
        self, profile: CircuitCostProfile, p: int
    ) -> tuple[tuple[int, ...], int]:
        """Cheap start, then improve by swapping one global qubit at a time."""
        n = self.model.num_qubits
        evaluated = 0

        # Start from the p qubits whose individual communication participation
        # is lowest. Scoring a singleton is not the same as scoring the set,
        # but it is a good ordering and costs n evaluations.
        singles = []
        for qubit in range(n):
            singles.append((profile.bytes_for((qubit,)), profile.exchanges_for((qubit,)), qubit))
            evaluated += 1
        singles.sort()
        current = tuple(sorted(q for _, _, q in singles[:p]))
        current_score = self._score(profile, current)
        evaluated += 1

        improved = True
        while improved:
            improved = False
            outside = [q for q in range(n) if q not in set(current)]
            for inside in current:
                for candidate_qubit in outside:
                    trial = tuple(sorted(set(current) - {inside} | {candidate_qubit}))
                    score = self._score(profile, trial)
                    evaluated += 1
                    if score < current_score:
                        current, current_score = trial, score
                        improved = True
                        break
                if improved:
                    break

        return current, evaluated


def optimize_placement(
    circuit: Circuit,
    world_size: int,
    precision: str = "fp64",
    candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
) -> MappingResult:
    """Convenience wrapper: build the cost model and run the mapper."""
    model = CommunicationCostModel(circuit.num_qubits, world_size, precision)
    return StaticCommunicationMapper(model, candidate_budget).optimize(circuit)
