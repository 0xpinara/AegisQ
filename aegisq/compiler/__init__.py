"""Communication-aware compilation: cost modelling and qubit placement."""

from aegisq.compiler.cost_model import (
    CommunicationCostModel,
    CommunicationEstimate,
    default_global_qubits,
    mapping_from_global_qubits,
)
from aegisq.compiler.fusion import (
    FusionResult,
    FusionStats,
    fuse,
    fuse_single_qubit_runs,
)
from aegisq.compiler.static_mapper import (
    MappingResult,
    StaticCommunicationMapper,
    optimize_placement,
)

__all__ = [
    "CommunicationCostModel",
    "CommunicationEstimate",
    "default_global_qubits",
    "mapping_from_global_qubits",
    "StaticCommunicationMapper",
    "MappingResult",
    "optimize_placement",
    "fuse",
    "fuse_single_qubit_runs",
    "FusionResult",
    "FusionStats",
]
