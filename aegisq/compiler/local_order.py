"""Choosing the order of the local qubits, which the byte model ignores.

The communication cost model only cares which qubits are *global*, because
that is what decides whether a gate sends anything. Everything below the
global cut is interchangeable as far as bytes go: two placements with the
same global set move byte for byte the same traffic. So the optimiser picks
a global set and leaves the local ordering at whatever the identity
permutation happened to give.

The kernel sweep says that ordering is not free in time. At 22 qubits on
eight threads, a `cz` on local position 1 runs at 22 GB/s and the same gate
on position 21 runs at 47 -- slower by a factor of two for no reason the
byte model can see. `h` and `rz` lean the other way, by less.

Since local ordering costs nothing in traffic, choosing it well is free.
This module does that: count how often each qubit is the position-sensitive
operand of each opcode, price every (qubit, position) pair from the measured
kernel data, and solve the resulting assignment problem exactly.

A caveat worth stating up front. The circuits in this suite are dominated by
`rz`, `cx` and `h`, whose position spread is much smaller than `cz`'s, so
the predicted saving is small -- single-digit percent of local kernel time
on most of them. Whether that survives the harness's noise floor is a
question for the benchmark, not for this docstring; `aegisq optimize
--local-order` prints the prediction and says it is a prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Positions the kernel sweep actually measures, at 22 qubits. Two
#: assumptions come with using them, and both are worth knowing about.
#:
#: Costs between anchors are interpolated linearly, and three points cannot
#: say whether the curve between them is straight. And the table is measured
#: at one circuit width but applied at all of them, so on a circuit much
#: narrower than 22 qubits every local position falls at or below the first
#: anchor and the model sees them as equally slow -- it will not order them
#: wrongly, it simply has nothing to say. Re-running the kernel sweep at the
#: width of interest is the fix; `load_position_costs` picks up whatever is
#: in benchmarks/raw.
MEASURED_POSITIONS = (1, 11, 21)

#: Seconds per gate at each measured position, from a run of
#: `aegisq benchmark kernels --qubits 22 --threads 8` on the reference host.
#: Used when no measured table is supplied. These are a fallback, not a
#: claim: `load_position_costs` reads the real numbers when they exist.
FALLBACK_POSITION_COSTS: dict[str, tuple[float, float, float]] = {
    "cz": (1.513e-3, 7.530e-4, 7.114e-4),
    "cx": (1.683e-3, 1.611e-3, 1.650e-3),
    "h": (1.759e-3, 1.766e-3, 2.120e-3),
    "rz": (1.670e-3, 1.650e-3, 1.803e-3),
    "swap": (1.598e-3, 1.507e-3, 1.583e-3),
}

#: Which operand's position the cost is attributed to. For the two-qubit
#: gates the sweep varies the target, so that is the operand priced here.
_COSTED_OPERAND = {"cx": 1, "cz": 1, "swap": 1}


@dataclass(frozen=True)
class LocalOrderResult:
    """A local ordering and what the model thinks it buys."""

    mapping: tuple[int, ...]
    """Full logical-to-physical permutation, global positions untouched."""

    local_order: tuple[int, ...]
    """Logical qubits in increasing local position."""

    baseline_seconds: float
    predicted_seconds: float

    @property
    def predicted_saving(self) -> float:
        """Fraction of predicted local kernel time removed, 0 if none."""
        if self.baseline_seconds <= 0:
            return 0.0
        return (self.baseline_seconds - self.predicted_seconds) / self.baseline_seconds


def load_position_costs(source=None) -> dict[str, tuple[float, ...]]:
    """Read per-opcode, per-position kernel costs from the raw sweep.

    Falls back to `FALLBACK_POSITION_COSTS` when no kernel measurements are
    present, so the optimiser still runs on a machine that has never been
    benchmarked -- with numbers from a different machine, which is why the
    caller is told which source was used.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a benchmark extra
        return dict(FALLBACK_POSITION_COSTS)

    from aegisq.benchmark.report import RAW_DIR

    directory = source or RAW_DIR
    paths = sorted(directory.glob("kernels_*.csv")) if directory.is_dir() else []
    if not paths:
        return dict(FALLBACK_POSITION_COSTS)

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.empty:
        return dict(FALLBACK_POSITION_COSTS)

    # The widest thread count available, since that is how the benchmark runs.
    frame = frame[frame["threads"] == frame["threads"].max()]
    costs: dict[str, tuple[float, ...]] = {}
    for kernel, group in frame.groupby("kernel"):
        by_position = group.groupby("target_qubit")["seconds_per_gate"].min().sort_index()
        if len(by_position) >= 2:
            costs[str(kernel)] = tuple(float(v) for v in by_position.to_numpy())
    return costs or dict(FALLBACK_POSITION_COSTS)


def _interpolate(costs: Sequence[float], positions: Sequence[int], position: int) -> float:
    """Cost at an arbitrary position, linear between measured anchors."""
    if position <= positions[0]:
        return float(costs[0])
    if position >= positions[-1]:
        return float(costs[-1])
    for index in range(len(positions) - 1):
        low, high = positions[index], positions[index + 1]
        if low <= position <= high:
            span = high - low
            weight = (position - low) / span if span else 0.0
            return float(costs[index] * (1 - weight) + costs[index + 1] * weight)
    return float(costs[-1])  # pragma: no cover - unreachable given the guards


def hungarian(cost: list[list[float]]) -> list[int]:
    """Minimum-cost assignment, exact, in O(n^3).

    Written out rather than imported: scipy would do this in one call but
    is not a dependency of the package, only of the benchmark extra, and a
    placement should not need the benchmark stack installed.

    This is the Jonker-Volgenant shortest-augmenting-path form. `cost` must
    be square. Returns `row -> column`.
    """
    size = len(cost)
    if size == 0:
        return []
    infinity = float("inf")
    # Potentials, with one extra slot used as the sentinel row/column.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    match = [0] * (size + 1)  # column -> row
    path = [0] * (size + 1)

    for row in range(1, size + 1):
        match[0] = row
        column = 0
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = match[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = cost[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    path[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[match[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if match[column] == 0:
                break
        while column:
            previous = path[column]
            match[column] = match[previous]
            column = previous

    assignment = [0] * size
    for candidate in range(1, size + 1):
        if match[candidate]:
            assignment[match[candidate] - 1] = candidate - 1
    return assignment


def _operand_counts(circuit, local_qubits: Sequence[int]) -> dict[int, dict[str, int]]:
    """How often each local qubit is the position-sensitive operand."""
    counts: dict[int, dict[str, int]] = {q: {} for q in local_qubits}
    for gate in circuit:
        operand_index = _COSTED_OPERAND.get(gate.opcode, 0)
        if operand_index >= len(gate.qubits):
            continue
        qubit = gate.qubits[operand_index]
        if qubit in counts:
            counts[qubit][gate.opcode] = counts[qubit].get(gate.opcode, 0) + 1
    return counts


def optimise_local_order(
    circuit,
    global_qubits: Sequence[int],
    num_qubits: int | None = None,
    position_costs: dict[str, tuple[float, ...]] | None = None,
) -> LocalOrderResult:
    """Assign local qubits to local positions to minimise predicted kernel time.

    Global qubits keep the positions the communication optimiser gave them,
    because moving those would change the traffic and that decision has
    already been made on better evidence. Only the local block is permuted,
    and permuting it is free in bytes.
    """
    num_qubits = num_qubits if num_qubits is not None else circuit.num_qubits
    costs = position_costs if position_costs is not None else load_position_costs()

    global_set = set(global_qubits)
    local_qubits = [q for q in range(num_qubits) if q not in global_set]
    local_positions = [p for p in range(num_qubits) if p not in global_set]

    identity = tuple(range(num_qubits))
    if len(local_qubits) < 2:
        return LocalOrderResult(identity, tuple(local_qubits), 0.0, 0.0)

    counts = _operand_counts(circuit, local_qubits)
    anchors = MEASURED_POSITIONS

    def price(qubit: int, position: int) -> float:
        total = 0.0
        for opcode, count in counts[qubit].items():
            table = costs.get(opcode)
            if table is None:
                continue
            spread = (
                anchors[: len(table)] if len(table) <= len(anchors) else tuple(range(len(table)))
            )
            total += count * _interpolate(table, spread, position)
        return total

    matrix = [[price(q, p) for p in local_positions] for q in local_qubits]
    assignment = hungarian(matrix)

    mapping = list(identity)
    order: list[tuple[int, int]] = []
    for index, qubit in enumerate(local_qubits):
        position = local_positions[assignment[index]]
        mapping[qubit] = position
        order.append((position, qubit))

    baseline = sum(price(q, local_positions[i]) for i, q in enumerate(local_qubits))
    predicted = sum(matrix[i][assignment[i]] for i in range(len(local_qubits)))
    return LocalOrderResult(
        mapping=tuple(mapping),
        local_order=tuple(q for _, q in sorted(order)),
        baseline_seconds=baseline,
        predicted_seconds=predicted,
    )
