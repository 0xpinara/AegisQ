"""Execution backends: reference (NumPy), native (C++) and distributed (MPI)."""

# Registering the native backend is a side effect of importing this module, so
# `Simulator(backend="cpp")` works without the caller knowing where it lives.
from aegisq.runtime import native as _native  # noqa: E402,F401  (import for side effect)
from aegisq.runtime.reference import ReferenceStateVector
from aegisq.runtime.simulator import (
    BackendUnavailableError,
    SimulationResult,
    Simulator,
    available_backends,
)

__all__ = [
    "ReferenceStateVector",
    "Simulator",
    "SimulationResult",
    "BackendUnavailableError",
    "available_backends",
]
