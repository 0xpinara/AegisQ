"""Bridge between the Python circuit IR and the compiled C++ core.

The native backend performs the same arithmetic as the reference backend but
with paired-index kernels and OpenMP threading. Both are validated against
each other in `tests/unit/test_native_backend.py`; when they disagree the
reference implementation is authoritative.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from aegisq import native_core
from aegisq.circuit.circuit import Circuit
from aegisq.runtime.simulator import BackendUnavailableError, register_backend

_PRECISION_CLASSES = {"fp64": "StateVectorF64", "fp32": "StateVectorF32"}


def require_core():
    """Return the native module or explain precisely how to obtain it."""
    core = native_core()
    if core is None:
        raise BackendUnavailableError(
            "the native core is not built; run `make build` (or ./scripts/build.sh) "
            "to compile the C++ extension"
        )
    return core


def to_native_circuit(circuit: Circuit):
    """Translate an AegisQ circuit into the C++ circuit representation."""
    core = require_core()
    native = core.Circuit(circuit.num_qubits)
    for gate in circuit:
        param = gate.params[0] if gate.params else 0.0
        native.add(core.Gate(gate.opcode, list(gate.qubits), param))
    return native


def new_state(num_qubits: int, precision: str = "fp64"):
    """Allocate a native state vector of the requested precision."""
    core = require_core()
    try:
        class_name = _PRECISION_CLASSES[precision]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; use 'fp64' or 'fp32'") from None
    return getattr(core, class_name)(num_qubits)


def counts_to_bitstrings(
    index_counts: dict[int, int],
    qubits: Sequence[int],
) -> dict[str, int]:
    """Fold basis-index counts into bitstring keys over ``qubits``.

    Marginalising after sampling the full index is equivalent to sampling the
    marginal distribution directly, so a partial measurement needs no separate
    sampling pass.
    """
    ordered = sorted(set(qubits), reverse=True)
    folded: dict[str, int] = {}
    for index, count in index_counts.items():
        key = "".join("1" if (index >> q) & 1 else "0" for q in ordered)
        folded[key] = folded.get(key, 0) + count
    return dict(sorted(folded.items()))


def _run_native(
    circuit: Circuit,
    *,
    shots: int,
    seed: int | None,
    precision: str,
    save_statevector: bool,
    measure: Sequence[int] | None,
    options: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, int], dict[str, Any]]:
    state = new_state(circuit.num_qubits, precision)
    native_circuit = to_native_circuit(circuit)

    started = time.perf_counter()
    state.apply_circuit(native_circuit)
    wall = time.perf_counter() - started

    counts: dict[str, int] = {}
    if shots:
        qubits = (
            measure
            if measure is not None
            else (circuit.measured_qubits or range(circuit.num_qubits))
        )
        raw = state.measure_all(shots, 0 if seed is None else int(seed))
        counts = counts_to_bitstrings(raw, list(qubits))

    amplitudes = np.array(state.amplitudes(), copy=True) if save_statevector else None
    return (
        amplitudes,
        counts,
        {
            "compute_seconds": state.compute_seconds,
            "kernel_wall_seconds": wall,
            "gates": int(state.gates_applied),
            "depth": circuit.depth(),
            "threads": require_core().max_threads(),
        },
    )


register_backend("cpp", _run_native)
