"""Fuse consecutive single-qubit gates into one unitary.

This is a second lever on communication volume, independent of where qubits
are placed.

A single-qubit gate on a *global* qubit costs a whole-shard exchange. Ten
consecutive such gates cost ten exchanges, even though their product is one
2x2 matrix. Multiplying the run out before execution turns ten exchanges into
one, and the arithmetic is identical because matrix multiplication is exactly
what the ten separate applications were computing anyway.

The pass is also useful on local qubits, where it trades ten strided sweeps
over the shard for one — a bandwidth saving rather than a network one.

Correctness
-----------
Single-qubit gates on different qubits commute, so a pending run may be
emitted later than it appeared as long as it stays ordered with respect to
every gate that touches its own qubit. Runs are therefore flushed exactly
when a two-qubit gate touches the qubit, and at the end of the circuit.

The fused matrix is carried in full, global phase included, so fusion changes
the state by nothing at all — not even a phase. That is worth the eight real
parameters: it means fused and unfused circuits can be compared amplitude by
amplitude rather than through a fidelity, and a discrepancy is always a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aegisq.circuit.circuit import Circuit
from aegisq.circuit.gates import GATE_SPECS, Gate, single_qubit_matrix, unitary_gate

#: Runs shorter than this are left alone. A single gate is already minimal,
#: and rewriting it as a fused matrix would only obscure it.
DEFAULT_MIN_RUN = 2


@dataclass
class FusionStats:
    """What the pass actually did, for reporting alongside measurements."""

    gates_before: int = 0
    gates_after: int = 0
    runs_fused: int = 0
    gates_absorbed: int = 0
    longest_run: int = 0
    diagonal_runs: int = 0

    @property
    def reduction(self) -> float:
        if not self.gates_before:
            return 0.0
        return (self.gates_before - self.gates_after) / self.gates_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "gates_before": self.gates_before,
            "gates_after": self.gates_after,
            "runs_fused": self.runs_fused,
            "gates_absorbed": self.gates_absorbed,
            "longest_run": self.longest_run,
            "diagonal_runs": self.diagonal_runs,
            "reduction": self.reduction,
        }

    def summary(self) -> str:
        return (
            f"fused {self.runs_fused} run(s) absorbing {self.gates_absorbed} gates: "
            f"{self.gates_before} -> {self.gates_after} gates "
            f"({self.reduction * 100:.1f}% fewer), longest run {self.longest_run}, "
            f"{self.diagonal_runs} run(s) stayed diagonal"
        )


@dataclass
class FusionResult:
    circuit: Circuit
    stats: FusionStats = field(default_factory=FusionStats)


def fuse_single_qubit_runs(
    circuit: Circuit,
    min_run: int = DEFAULT_MIN_RUN,
    name: str | None = None,
) -> FusionResult:
    """Return an equivalent circuit with single-qubit runs multiplied out."""
    if min_run < 2:
        raise ValueError("min_run must be at least 2; a single gate is already fused")

    fused = Circuit(circuit.num_qubits, name=name or f"{circuit.name}-fused")
    stats = FusionStats(gates_before=len(circuit))

    #: qubit -> (accumulated matrix, list of the gates it came from)
    pending: dict[int, tuple[np.ndarray, list[Gate]]] = {}

    def flush(qubit: int) -> None:
        entry = pending.pop(qubit, None)
        if entry is None:
            return
        matrix, sources = entry
        if len(sources) < min_run:
            for gate in sources:
                fused.append(gate)
            return

        gate = unitary_gate(qubit, matrix)
        fused.append(gate)
        stats.runs_fused += 1
        stats.gates_absorbed += len(sources)
        stats.longest_run = max(stats.longest_run, len(sources))
        if gate.is_diagonal:
            stats.diagonal_runs += 1

    for gate in circuit:
        spec = GATE_SPECS[gate.opcode]
        if spec.num_qubits == 1:
            qubit = gate.qubits[0]
            matrix = single_qubit_matrix(gate)
            if qubit in pending:
                accumulated, sources = pending[qubit]
                # The later gate acts on the result of the earlier one.
                pending[qubit] = (matrix @ accumulated, [*sources, gate])
            else:
                pending[qubit] = (np.array(matrix, dtype=complex), [gate])
            continue

        for operand in gate.qubits:
            flush(operand)
        fused.append(gate)

    for qubit in sorted(pending):
        flush(qubit)

    for qubit in circuit.measured_qubits:
        fused.measure(qubit)

    stats.gates_after = len(fused)
    return FusionResult(circuit=fused, stats=stats)


def fuse(circuit: Circuit, min_run: int = DEFAULT_MIN_RUN) -> Circuit:
    """Convenience wrapper returning just the fused circuit."""
    return fuse_single_qubit_runs(circuit, min_run=min_run).circuit
