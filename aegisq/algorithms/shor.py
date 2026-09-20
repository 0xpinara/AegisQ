"""Shor's algorithm at educational problem sizes.

**This factors small numbers such as 15 and 21. It is not, and does not
approach, a threat to RSA.** A cryptographically relevant modulus needs
thousands of logical qubits and error correction; simulating even a few dozen
qubits costs exponential memory, which is the entire premise of this project.
The point of including it is to show *why* post-quantum cryptography exists,
next to an implementation of the post-quantum primitives themselves.

How the quantum part is built
-----------------------------
Order finding needs a controlled modular multiplication `|y> -> |a*y mod N>`.
AegisQ has only one- and two-qubit gates, so the multiplication is synthesised
from first principles:

1. modular multiplication by `a` is a **permutation** of the work register's
   basis states (and the identity on states `>= N`);
2. a permutation decomposes into transpositions via its cycle structure;
3. a transposition `|x> <-> |y>` is realised by a CNOT ladder that reduces the
   difference between `x` and `y` to a single bit, a multi-controlled X on
   that bit, and the inverse ladder;
4. only the multi-controlled X needs the external control, because the ladder
   and its inverse cancel whenever the controlled gate does not fire.

This is general for any `N` that fits the work register, and it is exponential
in that register's width — which is exactly the honest reason it stops at toy
sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from aegisq.algorithms.grover import multi_controlled_z
from aegisq.algorithms.qft import controlled_phase
from aegisq.circuit.circuit import Circuit


def multi_controlled_x(
    circuit: Circuit, controls: list[int], target: int, ancillas: list[int]
) -> Circuit:
    """Flip `target` when every control is set, via H · MCZ · H."""
    if not controls:
        circuit.x(target)
        return circuit
    circuit.h(target)
    multi_controlled_z(circuit, controls, target, ancillas)
    circuit.h(target)
    return circuit


def controlled_transposition(
    circuit: Circuit,
    control: int,
    register: list[int],
    x: int,
    y: int,
    ancillas: list[int],
) -> Circuit:
    """Swap basis states `|x>` and `|y>` of `register`, conditioned on `control`."""
    if x == y:
        return circuit
    difference = x ^ y
    pivot_bit = (difference & -difference).bit_length() - 1
    other_bits = [b for b in range(len(register)) if (difference >> b) & 1 and b != pivot_bit]

    # Reduce the difference to the pivot bit alone. The ladder is applied
    # unconditionally; it is its own inverse and cancels when the controlled
    # gate below does not fire.
    for bit in other_bits:
        circuit.cx(register[pivot_bit], register[bit])

    # After the ladder both states share every bit except the pivot; that
    # shared pattern becomes the control condition.
    shared = x & ~difference
    for bit in other_bits:
        if (x >> bit) & 1:
            shared |= 1 << bit
        # the ladder has made x and y agree on `bit`; its value is x's, after
        # the flip that the ladder applied when the pivot bit of x is 1.
        if (x >> pivot_bit) & 1:
            shared ^= 1 << bit

    condition_bits = [b for b in range(len(register)) if b != pivot_bit]
    condition_qubits = [register[b] for b in condition_bits]
    # A control fires on |1>, so qubits whose required value is 0 are wrapped
    # in X gates.
    inverted = [register[b] for b in condition_bits if not (shared >> b) & 1]

    for qubit in inverted:
        circuit.x(qubit)
    multi_controlled_x(circuit, [control, *condition_qubits], register[pivot_bit], ancillas)
    for qubit in inverted:
        circuit.x(qubit)

    for bit in reversed(other_bits):
        circuit.cx(register[pivot_bit], register[bit])
    return circuit


def modular_multiplication_permutation(a: int, modulus: int, width: int) -> list[int]:
    """`|y> -> |a*y mod N>` as a permutation of `2^width` basis states.

    States `>= N` are left alone, which keeps the map a bijection.
    """
    if math.gcd(a, modulus) != 1:
        raise ValueError(f"{a} is not invertible modulo {modulus}")
    size = 1 << width
    if modulus > size:
        raise ValueError(f"work register of {width} qubits cannot hold values mod {modulus}")
    return [(a * y) % modulus if y < modulus else y for y in range(size)]


def permutation_transpositions(permutation: list[int]) -> list[tuple[int, int]]:
    """Transpositions realising a permutation, in the order they are applied.

    For a cycle `(c0 c1 ... ck)` the swaps `(c0,c1), (c0,c2), ..., (c0,ck)`
    applied in that order reproduce the cycle. Cycles are disjoint, so their
    relative order does not matter.
    """
    seen = [False] * len(permutation)
    transpositions: list[tuple[int, int]] = []
    for start in range(len(permutation)):
        if seen[start] or permutation[start] == start:
            seen[start] = True
            continue
        cycle = []
        node = start
        while not seen[node]:
            seen[node] = True
            cycle.append(node)
            node = permutation[node]
        for index in range(1, len(cycle)):
            transpositions.append((cycle[0], cycle[index]))
    return transpositions


def controlled_modular_multiplication(
    circuit: Circuit,
    control: int,
    work: list[int],
    ancillas: list[int],
    a: int,
    modulus: int,
) -> Circuit:
    """Append a controlled `|y> -> |a*y mod N>`."""
    permutation = modular_multiplication_permutation(a, modulus, len(work))
    for x, y in permutation_transpositions(permutation):
        controlled_transposition(circuit, control, work, x, y, ancillas)
    return circuit


def inverse_qft_on(circuit: Circuit, qubits: list[int]) -> Circuit:
    """Inverse QFT applied in place to an explicit list of qubits.

    The exact inverse of `aegisq.algorithms.qft.qft`: undo the bit-reversal
    swaps first, then walk the qubits in the opposite order with negated
    phases.
    """
    count = len(qubits)
    for index in range(count // 2):
        circuit.swap(qubits[index], qubits[count - 1 - index])
    for j in range(count):
        for k in range(j):
            controlled_phase(circuit, qubits[k], qubits[j], -math.pi / (2 ** (j - k)))
        circuit.h(qubits[j])
    return circuit


@dataclass
class ShorCircuit:
    """The circuit plus the register layout needed to read its output."""

    circuit: Circuit
    counting_qubits: list[int]
    work_qubits: list[int]
    ancilla_qubits: list[int]
    modulus: int
    base: int

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits


def shor_circuit(
    modulus: int,
    base: int,
    counting_qubits: int | None = None,
) -> ShorCircuit:
    """Order-finding circuit for `base` modulo `modulus`."""
    if modulus < 3 or modulus % 2 == 0:
        raise ValueError("this demonstration handles odd composites such as 15 and 21")
    if math.gcd(base, modulus) != 1:
        raise ValueError(f"gcd({base}, {modulus}) != 1; that case is solved classically")

    work_width = modulus.bit_length()
    counting = counting_qubits or 2 * work_width
    # A multi-controlled X over (counting control + work_width - 1) controls
    # needs that many minus one ancillas for the Toffoli ladder.
    ancilla_count = max(0, work_width - 1)

    total = counting + work_width + ancilla_count
    counting_qubits_list = list(range(counting))
    work = list(range(counting, counting + work_width))
    ancillas = list(range(counting + work_width, total))

    circuit = Circuit(total, name=f"shor{modulus}a{base}")
    for qubit in counting_qubits_list:
        circuit.h(qubit)
    circuit.x(work[0])  # work register starts at |1>

    # Repeated squaring: the j-th counting qubit controls multiplication by
    # base^(2^j) mod N, which is one controlled permutation rather than 2^j of
    # them.
    for index, control in enumerate(counting_qubits_list):
        factor = pow(base, 1 << index, modulus)
        controlled_modular_multiplication(circuit, control, work, ancillas, factor, modulus)

    inverse_qft_on(circuit, counting_qubits_list)
    for qubit in counting_qubits_list:
        circuit.measure(qubit)

    return ShorCircuit(
        circuit=circuit,
        counting_qubits=counting_qubits_list,
        work_qubits=work,
        ancilla_qubits=ancillas,
        modulus=modulus,
        base=base,
    )


def period_from_measurement(value: int, counting_qubits: int, modulus: int) -> int | None:
    """Recover a candidate order from one phase measurement.

    The counting register encodes `s/r` for some integer `s`; continued
    fractions recover `r` from the measured approximation.
    """
    if value == 0:
        return None
    phase = Fraction(value, 1 << counting_qubits)
    candidate = phase.limit_denominator(modulus).denominator
    return candidate if candidate > 1 else None


@dataclass
class ShorResult:
    """Outcome of the full hybrid procedure."""

    modulus: int
    base: int
    order: int | None
    factors: tuple[int, int] | None
    counting_qubits: int
    shots: int
    measurements: dict[str, int] = field(default_factory=dict)
    candidate_orders: dict[int, int] = field(default_factory=dict)
    note: str = ""

    @property
    def succeeded(self) -> bool:
        return self.factors is not None

    def summary(self) -> str:
        lines = [
            f"Shor's algorithm on N = {self.modulus} (educational size)",
            f"  base a:            {self.base}",
            f"  counting qubits:   {self.counting_qubits}",
            f"  shots:             {self.shots}",
        ]
        if self.candidate_orders:
            top = sorted(self.candidate_orders.items(), key=lambda kv: -kv[1])[:5]
            lines.append(
                "  candidate orders:  "
                + ", ".join(f"r={order} ({count} shots)" for order, count in top)
            )
        if self.order is not None:
            lines.append(f"  recovered order:   r = {self.order}")
            half = pow(self.base, self.order // 2, self.modulus)
            lines.append(f"  a^(r/2) mod N:     {half}")
        if self.factors is not None:
            p, q = self.factors
            lines.append(f"  factors:           {self.modulus} = {p} x {q}")
        else:
            lines.append(f"  no factorisation from this run: {self.note}")
        lines.append("")
        lines.append("  This demonstrates the algorithm at a size a laptop can simulate. It is")
        lines.append("  not a claim about factoring cryptographically relevant RSA moduli.")
        return "\n".join(lines)


def factors_from_order(modulus: int, base: int, order: int) -> tuple[int, int] | None:
    """The classical step: turn an even order into a factor pair."""
    if order % 2 != 0:
        return None
    root = pow(base, order // 2, modulus)
    if root == modulus - 1:
        return None
    candidates = (math.gcd(root - 1, modulus), math.gcd(root + 1, modulus))
    for candidate in candidates:
        if 1 < candidate < modulus and modulus % candidate == 0:
            return (candidate, modulus // candidate)
    return None


def factor(
    modulus: int = 15,
    shots: int = 1024,
    seed: int = 42,
    counting_qubits: int | None = None,
    bases: list[int] | None = None,
    backend: str = "cpp",
) -> ShorResult:
    """The full procedure: try bases until one yields a factorisation.

    Mirrors how Shor's algorithm is actually used — an unlucky base gives an
    odd order or a trivial square root, and the classical wrapper simply
    tries another one.
    """
    if modulus % 2 == 0:
        raise ValueError("even numbers are factored classically; use an odd composite")
    candidates = bases or [a for a in range(2, modulus) if math.gcd(a, modulus) == 1]
    attempts: list[ShorResult] = []
    for index, base in enumerate(candidates):
        common = math.gcd(base, modulus)
        if common > 1:  # pragma: no cover - filtered above, kept for clarity
            continue
        result = run_shor(
            modulus,
            base,
            shots=shots,
            seed=seed + index,
            counting_qubits=counting_qubits,
            backend=backend,
        )
        attempts.append(result)
        if result.succeeded:
            return result
    return (
        attempts[-1]
        if attempts
        else ShorResult(
            modulus=modulus,
            base=0,
            order=None,
            factors=None,
            counting_qubits=0,
            shots=shots,
            note="no usable base found",
        )
    )


def run_shor(
    modulus: int = 15,
    base: int = 7,
    shots: int = 2048,
    seed: int = 42,
    counting_qubits: int | None = None,
    backend: str = "cpp",
) -> ShorResult:
    """Run the order-finding circuit and complete the classical post-processing."""
    from aegisq.runtime import Simulator

    built = shor_circuit(modulus, base, counting_qubits)
    counting = len(built.counting_qubits)

    result = Simulator(backend).run(built.circuit, shots=shots, seed=seed)

    # Counts are keyed with the highest measured qubit first; the counting
    # register's value reads the other way round.
    candidates: dict[int, int] = {}
    for bitstring, count in result.counts.items():
        value = int(bitstring, 2)
        order = period_from_measurement(value, counting, modulus)
        if order is None:
            continue
        candidates[order] = candidates.get(order, 0) + count

    order = None
    factors = None
    note = "no candidate order produced a non-trivial factor"
    # Prefer the smallest order consistent with the measurements: continued
    # fractions can return a multiple of the true order, and a multiple is
    # less likely to survive the classical step.
    valid = sorted(r for r in candidates if pow(base, r, modulus) == 1)
    for candidate in valid:
        found = factors_from_order(modulus, base, candidate)
        if found is not None:
            order, factors = candidate, found
            note = ""
            break
        if order is None:
            order = candidate
            note = (
                f"order r = {candidate} is odd or gives a trivial square root; "
                "the classical procedure would retry with a different base"
            )

    return ShorResult(
        modulus=modulus,
        base=base,
        order=order,
        factors=factors,
        counting_qubits=counting,
        shots=shots,
        measurements=result.counts,
        candidate_orders=candidates,
        note=note,
    )
