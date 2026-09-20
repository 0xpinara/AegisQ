"""Distributed (MPI) execution helpers.

MPI is initialised lazily: importing this module is free, and the world
communicator only comes up when a caller actually asks for a rank, a world
size or a distributed state. That keeps `import aegisq` usable inside plain
Python processes, CI jobs and the packaging tests.
"""

from __future__ import annotations

from typing import Any

from aegisq import native_core
from aegisq.runtime.simulator import BackendUnavailableError

_PRECISION_CLASSES = {
    "fp64": "DistributedStateVectorF64",
    "fp32": "DistributedStateVectorF32",
}


def _core():
    core = native_core()
    if core is None:
        raise BackendUnavailableError(
            "the native core is not built; run `make build` to compile the C++ extension"
        )
    return core


def mpi_compiled() -> bool:
    """True when the native core was built against an MPI implementation."""
    core = native_core()
    return bool(core is not None and core.mpi_compiled())


# MPI_Finalize is deliberately *not* registered here.
#
# The native core calls `std::atexit(&MpiContext::finalize)` from the same
# place it calls `MPI_Init_thread`, so the two are registered together and
# only when this process is the one that initialised MPI. A second
# registration on the Python side was redundant and, worse, earlier:
# Python's `atexit` handlers run during interpreter shutdown, ahead of the
# C-level ones, so MPI could be finalised while objects that still refer to
# the MPI world were waiting to be collected. MPICH treats an MPI call after
# `MPI_Finalize` as fatal and aborts the rank, which is one candidate
# explanation for this suite's intermittent MPICH failures; Open MPI is more
# forgiving, and passes.
#
# Finalising only what we initialised, at the last possible moment, is the
# correct lifecycle regardless of whether it turns out to be that bug.


def world_size() -> int:
    """Number of ranks in the world communicator (1 without MPI)."""
    if not mpi_compiled():
        return 1
    return int(_core().mpi_world_size())


def rank() -> int:
    """Rank of this process (0 without MPI)."""
    if not mpi_compiled():
        return 0
    return int(_core().mpi_rank())


def is_distributed() -> bool:
    """True when this process is one of several cooperating ranks."""
    return world_size() > 1


def barrier() -> None:
    if mpi_compiled():
        _core().mpi_barrier()


def mpi_library_version() -> str:
    core = native_core()
    return str(core.mpi_library_version()) if core is not None else "none"


def layout(
    num_qubits: int,
    world_size_override: int | None = None,
    rank_override: int | None = None,
    mapping: list[int] | None = None,
):
    """Build a :class:`DistributedLayout`, defaulting to the live MPI world.

    Passing explicit ``world_size_override``/``rank_override`` makes the layout
    arithmetic testable in a serial process, which is how the placement rules
    are unit tested without launching mpirun.
    """
    core = _core()
    size = world_size() if world_size_override is None else world_size_override
    this_rank = rank() if rank_override is None else rank_override
    if mapping is None:
        return core.DistributedLayout(num_qubits, size, this_rank)
    return core.DistributedLayout(num_qubits, size, this_rank, list(mapping))


def new_distributed_state(
    num_qubits: int,
    precision: str = "fp64",
    mapping: list[int] | None = None,
) -> Any:
    """Allocate the rank-local shard of a distributed state vector."""
    core = _core()
    if not mpi_compiled():
        raise BackendUnavailableError(
            "the native core was built without MPI; rebuild with -DAEGISQ_ENABLE_MPI=ON"
        )
    try:
        class_name = _PRECISION_CLASSES[precision]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; use 'fp64' or 'fp32'") from None
    cls = getattr(core, class_name)
    if mapping is None:
        return cls(num_qubits)
    return cls(num_qubits, list(mapping))


def _run_distributed(
    circuit,
    *,
    shots: int,
    seed: int | None,
    precision: str,
    save_statevector: bool,
    measure,
    options: dict[str, Any],
):
    """Simulator backend: execute a circuit across the MPI world."""
    import time

    from aegisq.runtime.native import counts_to_bitstrings, to_native_circuit

    mapping = options.get("mapping")
    state = new_distributed_state(circuit.num_qubits, precision=precision, mapping=mapping)

    started = time.perf_counter()
    state.apply_circuit(to_native_circuit(circuit))
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

    statevector = None
    if save_statevector:
        # Gathering materialises 2^n amplitudes on every rank, so it is
        # allowed only for circuits small enough that this is harmless.
        limit = int(options.get("max_gather_qubits", 24))
        if circuit.num_qubits > limit:
            raise ValueError(
                f"refusing to gather a {circuit.num_qubits}-qubit state onto every rank; "
                f"raise max_gather_qubits above {limit} if this is intentional"
            )
        statevector = state.gather()

    metrics = dict(state.reduced_metrics())
    metrics.update(
        {
            "wall_seconds_rank": wall,
            "world_size": state.world_size,
            "local_amplitudes": state.local_size,
            "gates": len(circuit),
            "depth": circuit.depth(),
            "global_qubits": list(state.layout.global_qubits()),
            "mapping": list(state.layout.logical_to_position),
        }
    )
    return statevector, counts, metrics


def register_mpi_backend() -> None:
    """Register the `mpi` backend with the simulator front end."""
    from aegisq.runtime.simulator import available_backends, register_backend

    if "mpi" not in available_backends():
        register_backend("mpi", _run_distributed)


register_mpi_backend()


def preferred_backend() -> str:
    """`mpi` when this process is part of a multi-rank world, else `cpp`."""
    return "mpi" if is_distributed() else "cpp"


def describe() -> dict[str, Any]:
    """Short description of the distributed environment, for provenance records."""
    return {
        "mpi_compiled": mpi_compiled(),
        "mpi_library": mpi_library_version(),
        "world_size": world_size(),
        "rank": rank(),
    }
