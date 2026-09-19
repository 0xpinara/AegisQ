"""Distributed (MPI) execution helpers.

MPI is initialised lazily: importing this module is free, and the world
communicator only comes up when a caller actually asks for a rank, a world
size or a distributed state. That keeps `import aegisq` usable inside plain
Python processes, CI jobs and the packaging tests.
"""

from __future__ import annotations

import atexit
from typing import Any

from aegisq import native_core
from aegisq.runtime.simulator import BackendUnavailableError

_PRECISION_CLASSES = {
    "fp64": "DistributedStateVectorF64",
    "fp32": "DistributedStateVectorF32",
}

_finalize_registered = False


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


def _ensure_finalize_registered() -> None:
    global _finalize_registered
    if not _finalize_registered:
        atexit.register(lambda: native_core().mpi_finalize())
        _finalize_registered = True


def world_size() -> int:
    """Number of ranks in the world communicator (1 without MPI)."""
    if not mpi_compiled():
        return 1
    _ensure_finalize_registered()
    return int(_core().mpi_world_size())


def rank() -> int:
    """Rank of this process (0 without MPI)."""
    if not mpi_compiled():
        return 0
    _ensure_finalize_registered()
    return int(_core().mpi_rank())


def is_distributed() -> bool:
    """True when this process is one of several cooperating ranks."""
    return world_size() > 1


def barrier() -> None:
    if mpi_compiled():
        _ensure_finalize_registered()
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
    _ensure_finalize_registered()
    cls = getattr(core, class_name)
    if mapping is None:
        return cls(num_qubits)
    return cls(num_qubits, list(mapping))


def describe() -> dict[str, Any]:
    """Short description of the distributed environment, for provenance records."""
    return {
        "mpi_compiled": mpi_compiled(),
        "mpi_library": mpi_library_version(),
        "world_size": world_size(),
        "rank": rank(),
    }
