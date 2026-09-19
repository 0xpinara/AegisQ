"""Deliberately simple NumPy state-vector simulator.

This backend is the correctness oracle for everything else in AegisQ. It is
written for readability, not speed: the C++ and MPI engines are validated
against it, so a bug hidden behind a clever optimisation here would propagate
silently into every other backend.

Index convention
----------------
The state is a flat array of ``2**n`` amplitudes where basis index ``i`` has
bit ``(i >> q) & 1`` for qubit ``q``. Reshaped to ``(2,) * n`` in C order,
tensor axis ``a`` therefore corresponds to qubit ``n - 1 - a``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aegisq.circuit.circuit import Circuit
from aegisq.circuit.gates import Gate, single_qubit_matrix, two_qubit_matrix
from aegisq.circuit.validation import CircuitError, validate_gate

#: Largest state the reference backend will allocate without an explicit
#: override (2**28 complex128 = 4 GiB); guards against an accidental 2**40.
DEFAULT_MAX_QUBITS = 28

_DTYPES = {"fp64": np.complex128, "fp32": np.complex64}


def dtype_for(precision: str) -> np.dtype:
    try:
        return np.dtype(_DTYPES[precision])
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; use 'fp64' or 'fp32'") from None


class ReferenceStateVector:
    """Full state vector held in one process."""

    def __init__(
        self,
        num_qubits: int,
        precision: str = "fp64",
        max_qubits: int = DEFAULT_MAX_QUBITS,
    ) -> None:
        if num_qubits > max_qubits:
            raise MemoryError(
                f"refusing to allocate a {num_qubits}-qubit reference state "
                f"({2**num_qubits * np.dtype(dtype_for(precision)).itemsize / 2**30:.1f} GiB); "
                f"raise max_qubits explicitly if this is intentional"
            )
        self.num_qubits = num_qubits
        self.precision = precision
        self._dtype = dtype_for(precision)
        self._psi = np.zeros(2**num_qubits, dtype=self._dtype)
        self._psi[0] = 1.0

    # -- state access -------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Flat amplitude array (a view; copy before mutating)."""
        return self._psi

    def set_state(self, amplitudes: Sequence[complex]) -> None:
        vector = np.asarray(amplitudes, dtype=self._dtype)
        if vector.shape != (2**self.num_qubits,):
            raise ValueError(f"expected {2**self.num_qubits} amplitudes, got {vector.shape}")
        self._psi = vector.copy()

    def reset(self) -> None:
        self._psi[:] = 0.0
        self._psi[0] = 1.0

    def norm(self) -> float:
        """Squared 2-norm, which must stay at 1 under unitary evolution."""
        return float(np.vdot(self._psi, self._psi).real)

    # -- gate application ---------------------------------------------------

    def _axis(self, qubit: int) -> int:
        return self.num_qubits - 1 - qubit

    def apply(self, gate: Gate) -> None:
        gate = validate_gate(gate, self.num_qubits)
        if gate.spec.num_qubits == 1:
            self._apply_one(single_qubit_matrix(gate), gate.qubits[0])
        elif gate.spec.num_qubits == 2:
            self._apply_two(two_qubit_matrix(gate), gate.qubits[0], gate.qubits[1])
        else:  # pragma: no cover - no 3-qubit gates in the supported set
            raise CircuitError(f"cannot apply {gate.opcode}: unsupported arity")

    def _apply_one(self, matrix: np.ndarray, qubit: int) -> None:
        axis = self._axis(qubit)
        tensor = self._psi.reshape((2,) * self.num_qubits)
        out = np.tensordot(matrix.astype(self._dtype), tensor, axes=([1], [axis]))
        self._psi = np.moveaxis(out, 0, axis).reshape(-1)

    def _apply_two(self, matrix: np.ndarray, qubit_a: int, qubit_b: int) -> None:
        axis_a, axis_b = self._axis(qubit_a), self._axis(qubit_b)
        tensor = self._psi.reshape((2,) * self.num_qubits)
        op = matrix.astype(self._dtype).reshape(2, 2, 2, 2)
        out = np.tensordot(op, tensor, axes=([2, 3], [axis_a, axis_b]))
        self._psi = np.moveaxis(out, [0, 1], [axis_a, axis_b]).reshape(-1)

    def apply_circuit(self, circuit: Circuit) -> None:
        if circuit.num_qubits != self.num_qubits:
            raise CircuitError(
                f"circuit has {circuit.num_qubits} qubits, state has {self.num_qubits}"
            )
        for gate in circuit:
            self.apply(gate)

    # -- measurement --------------------------------------------------------

    def probabilities(self, qubits: Sequence[int] | None = None) -> np.ndarray:
        """Outcome distribution over ``qubits`` (all qubits when ``None``).

        The returned array is indexed so that the *highest* requested qubit is
        the most significant bit, matching the bitstring keys produced by
        :meth:`sample`.
        """
        probs = np.abs(self._psi) ** 2
        if qubits is None or sorted(qubits) == list(range(self.num_qubits)):
            return probs.astype(np.float64)

        selected = sorted(set(qubits), reverse=True)
        for q in selected:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} out of range")
        tensor = probs.reshape((2,) * self.num_qubits)
        keep_axes = [self._axis(q) for q in selected]  # ascending axes
        sum_axes = tuple(a for a in range(self.num_qubits) if a not in keep_axes)
        marginal = tensor.sum(axis=sum_axes) if sum_axes else tensor
        return marginal.reshape(-1).astype(np.float64)

    def sample(
        self,
        shots: int,
        seed: int | None = None,
        qubits: Sequence[int] | None = None,
    ) -> dict[str, int]:
        """Sample measurement outcomes, returning ``{bitstring: count}``.

        Sampling with the same seed, shots and circuit is bit-for-bit
        reproducible, which the provenance layer relies on.
        """
        if shots < 0:
            raise ValueError("shots must be non-negative")
        if shots == 0:
            return {}

        probs = self.probabilities(qubits)
        total = probs.sum()
        if total <= 0:
            raise RuntimeError("state has zero norm; cannot sample")
        probs = probs / total

        width = int(np.log2(probs.size))
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(shots, probs)
        return {
            format(index, f"0{width}b"): int(count) for index, count in enumerate(draws) if count
        }


def simulate(
    circuit: Circuit,
    shots: int = 0,
    seed: int | None = None,
    precision: str = "fp64",
) -> tuple[np.ndarray, dict[str, int]]:
    """Convenience helper: run ``circuit`` and return ``(statevector, counts)``."""
    state = ReferenceStateVector(circuit.num_qubits, precision=precision)
    state.apply_circuit(circuit)
    measured = circuit.measured_qubits or None
    counts = state.sample(shots, seed, measured) if shots else {}
    return state.state.copy(), counts
