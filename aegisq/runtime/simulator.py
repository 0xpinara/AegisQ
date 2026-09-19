"""Backend-agnostic simulation front end.

``Simulator`` is the single entry point used by the CLI, the algorithms module
and the benchmark runner. Backends are registered here as they are
implemented, so an unavailable backend fails with a precise message instead of
silently falling back to a slower one — a silent fallback would corrupt
benchmark results.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aegisq.circuit.circuit import Circuit


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one circuit execution."""

    counts: dict[str, int]
    num_qubits: int
    shots: int
    seed: int | None
    backend: str
    precision: str
    statevector: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_bits(self) -> int:
        return len(next(iter(self.counts))) if self.counts else 0

    def probabilities(self) -> dict[str, float]:
        if not self.shots:
            return {}
        return {key: value / self.shots for key, value in self.counts.items()}

    def most_frequent(self, top: int = 1) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend exists but cannot run on this machine."""


#: backend name -> factory(precision, options) -> runner callable
_BACKENDS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


class Simulator:
    """Execute circuits on a selected backend.

    Parameters
    ----------
    backend:
        ``"reference"`` is the NumPy oracle. Faster and distributed backends
        are registered by later layers of the package.
    precision:
        ``"fp64"`` (complex128) or ``"fp32"`` (complex64).
    """

    def __init__(self, backend: str = "reference", precision: str = "fp64", **options: Any) -> None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; available: {', '.join(available_backends())}"
            )
        if precision not in ("fp64", "fp32"):
            raise ValueError(f"unknown precision {precision!r}; use 'fp64' or 'fp32'")
        self.backend = backend
        self.precision = precision
        self.options = options

    def run(
        self,
        circuit: Circuit,
        shots: int = 0,
        seed: int | None = None,
        save_statevector: bool | None = None,
        measure: Sequence[int] | None = None,
    ) -> SimulationResult:
        """Run ``circuit`` and return counts and optionally the final state.

        ``save_statevector`` defaults to True when no shots are requested, so
        ``run(circuit)`` alone gives the amplitudes used by correctness tests.
        """
        if save_statevector is None:
            save_statevector = shots == 0
        runner = _BACKENDS[self.backend]
        started = time.perf_counter()
        statevector, counts, metrics = runner(
            circuit,
            shots=shots,
            seed=seed,
            precision=self.precision,
            save_statevector=save_statevector,
            measure=measure,
            options=self.options,
        )
        elapsed = time.perf_counter() - started
        metrics = {"wall_seconds": elapsed, **metrics}
        return SimulationResult(
            counts=counts,
            num_qubits=circuit.num_qubits,
            shots=shots,
            seed=seed,
            backend=self.backend,
            precision=self.precision,
            statevector=statevector,
            metrics=metrics,
        )


def _run_reference(
    circuit: Circuit,
    *,
    shots: int,
    seed: int | None,
    precision: str,
    save_statevector: bool,
    measure: Sequence[int] | None,
    options: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, int], dict[str, Any]]:
    from aegisq.runtime.reference import ReferenceStateVector

    state = ReferenceStateVector(
        circuit.num_qubits,
        precision=precision,
        max_qubits=options.get("max_qubits", 28),
    )
    compute_started = time.perf_counter()
    state.apply_circuit(circuit)
    compute_seconds = time.perf_counter() - compute_started

    qubits = measure if measure is not None else (circuit.measured_qubits or None)
    counts = state.sample(shots, seed, qubits) if shots else {}
    return (
        state.state.copy() if save_statevector else None,
        counts,
        {
            "compute_seconds": compute_seconds,
            "gates": len(circuit),
            "depth": circuit.depth(),
        },
    )


register_backend("reference", _run_reference)
