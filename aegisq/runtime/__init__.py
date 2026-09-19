"""Execution backends: reference (NumPy), native (C++) and distributed (MPI)."""

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
