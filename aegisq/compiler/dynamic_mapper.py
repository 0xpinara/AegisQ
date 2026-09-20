"""Windowed placement: change the qubit assignment part-way through a circuit.

Static placement picks one assignment for the whole circuit. That is the right
answer when a circuit's communication structure is uniform, and the wrong one
when it is not: a circuit that works intensively on one group of qubits and
then moves to another wants a different assignment in each phase.

Re-assigning is not free. Moving a logical qubit between a local and a global
slot means physically moving amplitudes, at the cost of a half-shard exchange
per rank. So the decision at each window boundary is a comparison:

    cost(window under the current assignment)
        vs
    cost(window under a better assignment) + cost(getting there)

and the plan switches only when the second is smaller. That is the whole idea;
everything below is bookkeeping around it.

Execution without touching the runtime
--------------------------------------
A plan is applied as a **circuit rewrite**, not as a runtime feature. The
runtime keeps one fixed mapping, and the plan permutes which logical qubit
occupies which slot:

* gates in a window are rewritten to act on slots rather than logical qubits,
* a change of assignment becomes SWAP gates between slots, whose cost the
  existing profiler measures like any other gate,
* the final permutation is undone so the result is in logical order, and that
  restoration is paid for and counted rather than hidden.

The rewritten circuit therefore computes exactly the same state, and every
byte the plan spends is visible to the same instrumentation as everything
else.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from aegisq.circuit.circuit import Circuit
from aegisq.circuit.gates import Gate
from aegisq.compiler.cost_model import CommunicationCostModel, default_global_qubits

#: Gates per window. Small enough to track a change of phase, large enough
#: that a re-assignment can pay for itself.
DEFAULT_WINDOW_SIZE = 128

#: Candidate assignments kept per window when searching transitions. The full
#: transition search is quadratic in the candidate count, so the plan keeps
#: the best few per window rather than all C(n, p) of them.
DEFAULT_CANDIDATES_PER_WINDOW = 8


@dataclass(frozen=True)
class Window:
    """One slice of the circuit, with the assignments worth considering."""

    index: int
    start: int
    stop: int
    gates: tuple[Gate, ...]
    candidates: tuple[tuple[int, ...], ...]

    def __len__(self) -> int:
        return len(self.gates)


@dataclass
class DynamicPlan:
    """A placement per window, plus what it is predicted to cost."""

    windows: list[Window] = field(default_factory=list)
    assignments: list[tuple[int, ...]] = field(default_factory=list)
    transition_bytes: list[int] = field(default_factory=list)
    window_bytes: list[int] = field(default_factory=list)
    static_assignment: tuple[int, ...] = ()
    static_bytes: int = 0
    restore_bytes: int = 0
    window_size: int = DEFAULT_WINDOW_SIZE

    #: Logical qubit -> slot at the end of the plan. When the plan does not
    #: restore logical order in the circuit, results have to be relabelled by
    #: this permutation, which is free because it is classical bookkeeping.
    final_slots: tuple[int, ...] = ()
    restores_order: bool = False

    @property
    def dynamic_bytes(self) -> int:
        """Total predicted traffic, transitions and restoration included."""
        return sum(self.window_bytes) + sum(self.transition_bytes) + self.restore_bytes

    @property
    def switches(self) -> int:
        return sum(1 for cost in self.transition_bytes if cost > 0)

    @property
    def improvement(self) -> float:
        if not self.static_bytes:
            return 0.0
        return (self.static_bytes - self.dynamic_bytes) / self.static_bytes

    def worthwhile(self) -> bool:
        return self.dynamic_bytes < self.static_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "windows": len(self.windows),
            "assignments": [list(a) for a in self.assignments],
            "switches": self.switches,
            "static_assignment": list(self.static_assignment),
            "static_bytes": self.static_bytes,
            "dynamic_bytes": self.dynamic_bytes,
            "transition_bytes": list(self.transition_bytes),
            "restore_bytes": self.restore_bytes,
            "improvement": self.improvement,
        }

    def report(self) -> str:
        lines = [
            f"Windowed placement over {len(self.windows)} window(s) of {self.window_size} gates",
            f"  static assignment:   {list(self.static_assignment)} "
            f"-> {self.static_bytes / 2**20:.2f} MiB predicted",
            f"  windowed assignment: {self.switches} switch(es) "
            f"-> {self.dynamic_bytes / 2**20:.2f} MiB predicted "
            f"(transitions {sum(self.transition_bytes) / 2**20:.2f} MiB, "
            f"restoration {self.restore_bytes / 2**20:.2f} MiB)",
            f"  predicted change:    {self.improvement * 100:+.1f}%",
        ]
        if not self.worthwhile():
            lines.append("  the static assignment wins; applying this plan would cost more")
        return "\n".join(lines)


def _split_windows(circuit: Circuit, window_size: int) -> list[tuple[int, int]]:
    if window_size < 1:
        raise ValueError("window size must be at least one gate")
    bounds = []
    for start in range(0, len(circuit), window_size):
        bounds.append((start, min(start + window_size, len(circuit))))
    return bounds or [(0, 0)]


def _window_circuit(circuit: Circuit, gates: tuple[Gate, ...]) -> Circuit:
    window = Circuit(circuit.num_qubits, name="window")
    window.extend(gates)
    return window


def _best_assignments(
    model: CommunicationCostModel,
    window: Circuit,
    keep: int,
) -> list[tuple[int, ...]]:
    """The cheapest assignments for one window, best first."""
    p = model.num_global_qubits
    if p == 0:
        return [()]
    profile = model.compile_circuit(window)
    scored = [
        (profile.bytes_for(candidate), profile.exchanges_for(candidate), candidate)
        for candidate in itertools.combinations(range(model.num_qubits), p)
    ]
    scored.sort()
    return [candidate for _, _, candidate in scored[:keep]]


def swap_cost(model: CommunicationCostModel, slot_a: int, slot_b: int) -> int:
    """Bytes a SWAP between two physical slots costs, summed over ranks."""
    local = model.num_local_qubits
    if slot_a < local and slot_b < local:
        return 0
    return (model.shard_bytes // 2) * model.world_size


def transition_swaps(current: list[int], target: list[int]) -> list[tuple[int, int]]:
    """Slot pairs whose exchange turns one slot assignment into the other.

    Walks each cycle of the permutation, which is what `apply_plan` emits, so
    the planner and the rewriter cannot disagree about what a transition
    costs. They did once: costing only group changes missed the swaps between
    two global slots that a restoration needs, and the plan under-predicted
    its own traffic by exactly those exchanges.
    """
    slots = list(current)
    swaps: list[tuple[int, int]] = []
    for qubit in range(len(slots)):
        if slots[qubit] == target[qubit]:
            continue
        partner = next(
            other for other in range(len(slots)) if slots[other] == target[qubit] and other != qubit
        )
        swaps.append((slots[qubit], slots[partner]))
        slots[qubit], slots[partner] = slots[partner], slots[qubit]
    return swaps


def transition_cost(
    model: CommunicationCostModel,
    current: tuple[int, ...],
    target: tuple[int, ...],
) -> int:
    """Approximate bytes to move between two assignments, ignoring slot order.

    Each logical qubit that changes group trades slots with one changing the
    other way, and such a swap touches a global slot. This is what the search
    optimises over, because it depends only on the assignments and not on the
    path taken to reach them. The cost finally reported for a plan is computed
    exactly from the swaps it will actually emit.
    """
    moving = len(set(target) - set(current))
    return moving * (model.shard_bytes // 2) * model.world_size


def plan_dynamic_placement(
    circuit: Circuit,
    model: CommunicationCostModel,
    window_size: int = DEFAULT_WINDOW_SIZE,
    candidates_per_window: int = DEFAULT_CANDIDATES_PER_WINDOW,
    static_assignment: tuple[int, ...] | None = None,
    restore_order: bool = False,
) -> DynamicPlan:
    """Decide, window by window, whether re-assigning pays for itself.

    The transition search is a dynamic program over windows: each window keeps
    its few best assignments, and the plan is the cheapest path through them
    including the cost of every switch and of restoring logical order at the
    end. Restricting to a few candidates per window is what keeps it
    affordable; the restriction is a heuristic and is reported as such.

    The baseline is the best *static* assignment for the whole circuit, found
    by the ordinary placement search. Comparing against the default
    assignment instead would hand windowing the credit for a gain that static
    placement already delivers.

    `restore_order` decides who undoes the final permutation. With it off (the
    default) the plan stops wherever the last window left the qubits and the
    caller relabels its results, which costs nothing because it is classical
    bookkeeping — exactly what the runtime mapping does for a static
    assignment. Turning it on appends the SWAP gates to restore logical order
    inside the circuit and charges the plan for them, which is a real cost and
    makes windowing look worse than it is for no reason other than the choice
    of execution strategy.
    """
    if circuit.num_qubits != model.num_qubits:
        raise ValueError("circuit width does not match the cost model")

    plan = DynamicPlan(window_size=window_size)
    plan.restores_order = restore_order
    identity = tuple(default_global_qubits(model.num_qubits, model.world_size))

    if static_assignment is not None:
        plan.static_assignment = tuple(static_assignment)
    else:
        # The baseline is the *best* single assignment, not the default one.
        # Comparing against the default would credit windowing with the gain
        # that ordinary static placement already delivers on its own.
        from aegisq.compiler.static_mapper import StaticCommunicationMapper

        plan.static_assignment = StaticCommunicationMapper(model).optimize(circuit).global_qubits
    plan.static_bytes = model.estimate(circuit, plan.static_assignment).bytes_sent

    if model.num_global_qubits == 0 or not len(circuit):
        plan.windows = []
        plan.assignments = []
        plan.window_bytes = []
        plan.transition_bytes = []
        plan.final_slots = tuple(range(model.num_qubits))
        return plan

    windows: list[Window] = []
    for index, (start, stop) in enumerate(_split_windows(circuit, window_size)):
        gates = circuit.gates[start:stop]
        sub = _window_circuit(circuit, gates)
        candidates = _best_assignments(model, sub, candidates_per_window)
        # The static assignment is always available, so "do not switch" is
        # always one of the options.
        if plan.static_assignment not in candidates:
            candidates.append(plan.static_assignment)
        windows.append(
            Window(
                index=index,
                start=start,
                stop=stop,
                gates=gates,
                candidates=tuple(candidates),
            )
        )

    # Dynamic program: best[i][c] is the cheapest total cost of processing
    # windows 0..i with window i under assignment c.
    window_cost: list[dict[tuple[int, ...], int]] = []
    for window in windows:
        sub = _window_circuit(circuit, window.gates)
        profile = model.compile_circuit(sub)
        window_cost.append({c: profile.bytes_for(c) for c in window.candidates})

    best: dict[tuple[int, ...], int] = {}
    previous: list[dict[tuple[int, ...], tuple[int, ...] | None]] = []

    for index, window in enumerate(windows):
        current: dict[tuple[int, ...], int] = {}
        back: dict[tuple[int, ...], tuple[int, ...] | None] = {}
        for candidate in window.candidates:
            cost_here = window_cost[index][candidate]
            if index == 0:
                # Starting from the static assignment costs a transition too.
                current[candidate] = cost_here + transition_cost(
                    model, plan.static_assignment, candidate
                )
                back[candidate] = None
            else:
                options = (
                    (best[prior] + transition_cost(model, prior, candidate), prior)
                    for prior in best
                )
                total, chosen = min(options)
                current[candidate] = total + cost_here
                back[candidate] = chosen
        best = current
        previous.append(back)

    # When the plan restores logical order itself, that restoration is part of
    # the bill and the last window's assignment is chosen accordingly.
    if restore_order:
        final_candidate = min(best, key=lambda c: best[c] + transition_cost(model, c, identity))
    else:
        final_candidate = min(best, key=lambda c: best[c])

    # Walk the chain back to recover the assignment per window.
    chain: list[tuple[int, ...]] = [final_candidate]
    for index in range(len(windows) - 1, 0, -1):
        chain.append(previous[index][chain[-1]])
    chain.reverse()

    plan.windows = windows
    plan.assignments = chain
    plan.window_bytes = [window_cost[i][chain[i]] for i in range(len(windows))]

    # Replay the slot evolution to cost the transitions exactly, from the same
    # swap list the rewriter will emit. The search used an approximation; what
    # is reported must match what the runtime will actually send.
    slots = _slot_permutation(plan.static_assignment, model.num_qubits, model.num_local_qubits)
    plan.transition_bytes = []
    for assignment in chain:
        target = _align(slots, assignment, model.num_local_qubits)
        swaps = transition_swaps(slots, target)
        plan.transition_bytes.append(sum(swap_cost(model, a, b) for a, b in swaps))
        slots = target
    if restore_order:
        restore_swaps = transition_swaps(slots, list(range(model.num_qubits)))
        plan.restore_bytes = sum(swap_cost(model, a, b) for a, b in restore_swaps)
        plan.final_slots = tuple(range(model.num_qubits))
    else:
        plan.restore_bytes = 0
        plan.final_slots = tuple(slots)
    return plan


def _slot_permutation(assignment: tuple[int, ...], num_qubits: int, num_local: int) -> list[int]:
    """Logical qubit -> slot, putting `assignment` on the global slots."""
    chosen = sorted(assignment)
    locals_ = [q for q in range(num_qubits) if q not in set(chosen)]
    slots = [0] * num_qubits
    for position, qubit in enumerate(locals_):
        slots[qubit] = position
    for offset, qubit in enumerate(chosen):
        slots[qubit] = num_local + offset
    return slots


def _align(previous: list[int], target_globals: tuple[int, ...], num_local: int) -> list[int]:
    """Choose slots for `target_globals` that move as few qubits as possible."""
    num_qubits = len(previous)
    target = set(target_globals)
    slots = list(previous)

    staying_global = [q for q in range(num_qubits) if q in target and previous[q] >= num_local]
    staying_local = [q for q in range(num_qubits) if q not in target and previous[q] < num_local]
    entering = [q for q in range(num_qubits) if q in target and previous[q] < num_local]
    leaving = [q for q in range(num_qubits) if q not in target and previous[q] >= num_local]

    # Qubits that keep their group keep their slot; the rest trade in pairs.
    for arriving, departing in zip(entering, leaving, strict=True):
        slots[arriving], slots[departing] = slots[departing], slots[arriving]
    assert len(staying_global) + len(entering) == len(target_globals)
    assert len(staying_local) + len(leaving) == num_qubits - len(target_globals)
    return slots


def apply_plan(
    circuit: Circuit,
    plan: DynamicPlan,
    model: CommunicationCostModel,
) -> Circuit:
    """Rewrite a circuit so that it realises a windowed placement plan.

    Gates are re-expressed on slots, assignment changes become SWAP gates
    between slots, and the final permutation is undone so the result is in
    logical order.
    """
    num_qubits = circuit.num_qubits
    num_local = model.num_local_qubits
    rewritten = Circuit(num_qubits, name=f"{circuit.name}-windowed")

    if not plan.windows:
        rewritten.extend(circuit.gates)
        for qubit in circuit.measured_qubits:
            rewritten.measure(qubit)
        return rewritten

    slots = _slot_permutation(plan.static_assignment, num_qubits, num_local)

    def emit_transition(target_slots: list[int]) -> None:
        """Insert the SWAP gates that move the data into the new slots."""
        for slot_a, slot_b in transition_swaps(slots, target_slots):
            rewritten.swap(slot_a, slot_b)
        slots[:] = target_slots

    for window, assignment in zip(plan.windows, plan.assignments, strict=True):
        emit_transition(_align(slots, assignment, num_local))
        for gate in window.gates:
            operands = tuple(slots[q] for q in gate.qubits)
            rewritten.append(Gate(gate.opcode, operands, gate.params))

    if plan.restores_order:
        emit_transition(list(range(num_qubits)))

    # Measurement follows the qubit, so a measured logical qubit is read from
    # whichever slot it ended up in.
    for qubit in circuit.measured_qubits:
        rewritten.measure(slots[qubit])
    return rewritten


def permute_amplitudes(state, slots: tuple[int, ...]):
    """Relabel a state vector from slot order back to logical order.

    Free, classical bookkeeping: exactly what the runtime mapping does for a
    static assignment, which is why a windowed plan should not have to pay for
    it in SWAP gates.
    """
    import numpy as np

    state = np.asarray(state)
    num_qubits = len(slots)
    if state.shape != (1 << num_qubits,):
        raise ValueError(f"expected {1 << num_qubits} amplitudes, got {state.shape}")

    indices = np.arange(state.size)
    logical = np.zeros_like(indices)
    for qubit, slot in enumerate(slots):
        logical |= ((indices >> slot) & 1) << qubit
    permuted = np.empty_like(state)
    permuted[logical] = state
    return permuted


def permute_counts(
    counts: dict[str, int], slots: tuple[int, ...], measured: tuple[int, ...]
) -> dict[str, int]:
    """Relabel measurement counts from slot order back to logical order."""
    ordered_slots = sorted(slots[q] for q in measured)
    ordered_logical = sorted(measured, reverse=True)

    relabelled: dict[str, int] = {}
    for bitstring, count in counts.items():
        # The key is ordered by descending slot; rebuild it by descending
        # logical qubit instead.
        values = {
            slot: int(bit)
            for slot, bit in zip(sorted(ordered_slots, reverse=True), bitstring, strict=True)
        }
        key = "".join(str(values[slots[q]]) for q in ordered_logical)
        relabelled[key] = relabelled.get(key, 0) + count
    return dict(sorted(relabelled.items()))


def dynamic_placement(
    circuit: Circuit,
    world_size: int,
    precision: str = "fp64",
    window_size: int = DEFAULT_WINDOW_SIZE,
    candidates_per_window: int = DEFAULT_CANDIDATES_PER_WINDOW,
    restore_order: bool = False,
) -> tuple[Circuit, DynamicPlan]:
    """Plan and apply windowed placement, returning the rewritten circuit.

    If the plan does not beat the static assignment the circuit is returned
    unchanged, so calling this is never worse than not calling it.
    """
    model = CommunicationCostModel(circuit.num_qubits, world_size, precision)
    plan = plan_dynamic_placement(
        circuit,
        model,
        window_size=window_size,
        candidates_per_window=candidates_per_window,
        restore_order=restore_order,
    )
    if not plan.worthwhile():
        return circuit, plan
    return apply_plan(circuit, plan, model), plan
