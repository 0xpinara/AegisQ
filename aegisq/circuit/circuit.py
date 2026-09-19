"""Circuit container and builder.

Qubit ordering convention (used consistently across every AegisQ backend and
matching Qiskit's little-endian convention):

* qubit ``q`` contributes bit ``2**q`` to a computational-basis index,
* bitstrings printed by measurement put the highest-index measured qubit on
  the left, so a two-qubit key ``"10"`` means ``q1 = 1, q0 = 0``.

Measurement is terminal. Instructions are appended to a flat list; a
``measure`` simply records that a qubit is sampled at the end of the circuit.
Mid-circuit measurement and classical feed-forward are out of scope and are
rejected rather than silently reinterpreted.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from aegisq.circuit.gates import GATE_SPECS, Gate
from aegisq.circuit.validation import CircuitError, validate_gate, validate_num_qubits


@dataclass
class Circuit:
    """A sequence of gates over ``num_qubits`` qubits."""

    num_qubits: int
    name: str = "circuit"
    _gates: list[Gate] = field(default_factory=list, repr=False)
    _measured: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        validate_num_qubits(self.num_qubits)

    # -- construction -------------------------------------------------------

    def append(self, gate: Gate) -> Circuit:
        """Append a validated gate and return ``self`` for chaining."""
        self._gates.append(validate_gate(gate, self.num_qubits))
        return self

    def extend(self, gates: Iterable[Gate]) -> Circuit:
        for gate in gates:
            self.append(gate)
        return self

    def _one(self, opcode: str, qubit: int, *params: float) -> Circuit:
        return self.append(Gate(opcode, (qubit,), tuple(params)))

    def _two(self, opcode: str, a: int, b: int) -> Circuit:
        return self.append(Gate(opcode, (a, b)))

    def x(self, q: int) -> Circuit:
        return self._one("x", q)

    def y(self, q: int) -> Circuit:
        return self._one("y", q)

    def z(self, q: int) -> Circuit:
        return self._one("z", q)

    def h(self, q: int) -> Circuit:
        return self._one("h", q)

    def s(self, q: int) -> Circuit:
        return self._one("s", q)

    def t(self, q: int) -> Circuit:
        return self._one("t", q)

    def rx(self, q: int, theta: float) -> Circuit:
        return self._one("rx", q, theta)

    def ry(self, q: int, theta: float) -> Circuit:
        return self._one("ry", q, theta)

    def rz(self, q: int, theta: float) -> Circuit:
        return self._one("rz", q, theta)

    def cx(self, control: int, target: int) -> Circuit:
        return self._two("cx", control, target)

    def cz(self, a: int, b: int) -> Circuit:
        return self._two("cz", a, b)

    def swap(self, a: int, b: int) -> Circuit:
        return self._two("swap", a, b)

    def measure(self, qubit: int) -> Circuit:
        """Mark ``qubit`` for terminal measurement."""
        if not 0 <= qubit < self.num_qubits:
            raise CircuitError(f"cannot measure qubit {qubit} in a {self.num_qubits}-qubit circuit")
        if qubit not in self._measured:
            self._measured.append(qubit)
        return self

    def measure_all(self) -> Circuit:
        for q in range(self.num_qubits):
            self.measure(q)
        return self

    # -- inspection ---------------------------------------------------------

    @property
    def gates(self) -> tuple[Gate, ...]:
        return tuple(self._gates)

    @property
    def measured_qubits(self) -> tuple[int, ...]:
        """Measured qubits in ascending index order (empty means "sample all")."""
        return tuple(sorted(self._measured))

    def __len__(self) -> int:
        return len(self._gates)

    def __iter__(self) -> Iterator[Gate]:
        return iter(self._gates)

    def gate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for gate in self._gates:
            counts[gate.opcode] = counts.get(gate.opcode, 0) + 1
        return dict(sorted(counts.items()))

    def depth(self) -> int:
        """Circuit depth: longest chain of gates sharing a qubit."""
        frontier = [0] * self.num_qubits
        for gate in self._gates:
            level = max(frontier[q] for q in gate.qubits) + 1
            for q in gate.qubits:
                frontier[q] = level
        return max(frontier, default=0)

    def two_qubit_gate_count(self) -> int:
        return sum(1 for g in self._gates if GATE_SPECS[g.opcode].num_qubits == 2)

    def inverse(self) -> Circuit:
        """Circuit implementing the inverse unitary (measurements dropped)."""
        from aegisq.circuit.gates import inverse as gate_inverse

        out = Circuit(self.num_qubits, name=f"{self.name}_inv")
        for gate in reversed(self._gates):
            out.append(gate_inverse(gate))
        return out

    def compose(self, other: Circuit, name: str | None = None) -> Circuit:
        """Concatenate two circuits over the same register."""
        if other.num_qubits != self.num_qubits:
            raise CircuitError(
                f"cannot compose a {other.num_qubits}-qubit circuit into "
                f"a {self.num_qubits}-qubit circuit"
            )
        out = Circuit(self.num_qubits, name=name or self.name)
        out.extend(self._gates)
        out.extend(other._gates)
        for q in (*self.measured_qubits, *other.measured_qubits):
            out.measure(q)
        return out

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical dictionary form, used for hashing and job payloads."""
        return {
            "schema": "aegisq.circuit.v1",
            "name": self.name,
            "num_qubits": self.num_qubits,
            "gates": [
                {"opcode": g.opcode, "qubits": list(g.qubits), "params": list(g.params)}
                for g in self._gates
            ],
            "measured": list(self.measured_qubits),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Circuit:
        schema = payload.get("schema")
        if schema != "aegisq.circuit.v1":
            raise CircuitError(f"unknown circuit schema {schema!r}")
        circuit = cls(int(payload["num_qubits"]), name=str(payload.get("name", "circuit")))
        for item in payload["gates"]:
            circuit.append(
                Gate(
                    str(item["opcode"]),
                    tuple(int(q) for q in item["qubits"]),
                    tuple(float(p) for p in item.get("params", ())),
                )
            )
        for q in payload.get("measured", ()):
            circuit.measure(int(q))
        return circuit

    def __str__(self) -> str:
        head = f"Circuit({self.name}, qubits={self.num_qubits}, gates={len(self._gates)}, depth={self.depth()})"
        body = "\n".join(f"  {g}" for g in self._gates[:20])
        if len(self._gates) > 20:
            body += f"\n  ... {len(self._gates) - 20} more"
        return f"{head}\n{body}" if body else head


def qubit_bitstring(index: int, qubits: Sequence[int]) -> str:
    """Render basis-state ``index`` over ``qubits`` with the highest index left."""
    return "".join("1" if (index >> q) & 1 else "0" for q in sorted(qubits, reverse=True))
