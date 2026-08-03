# schemas/__init__.py
"""Shared domain schemas for the DLens framework.

V1 simulation-stage schemas (the ``sim_config`` / ``sim_output`` slots of an
experiment run) plus the V2 code-generation schemas.
"""

from dlens.schemas._codegen import CodegenResult, SimSpec, ValidationResult
from dlens.schemas._simulation import (
    CosmologyParams,
    SimConfig,
    SimModelConfig,
    SimOutput,
    SubstructureType,
)

__all__ = [
    "SubstructureType",
    "SimModelConfig",
    "CosmologyParams",
    "SimConfig",
    "SimOutput",
    "SimSpec",
    "ValidationResult",
    "CodegenResult",
]
