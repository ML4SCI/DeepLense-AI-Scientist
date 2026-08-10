# schemas/__init__.py
"""Shared domain schemas for the DLens framework.

V1 simulation-stage schemas (the ``sim_config`` / ``sim_output`` slots of an
experiment run), the V2 code-generation schemas, and the two-stage parameter
extraction/validation schemas.
"""

from dlens.schemas._codegen import CodegenResult, SimSpec, ValidationResult
from dlens.schemas._lens_params import (
    CodeParamComparison,
    FieldComparison,
    InstrumentBlock,
    LensParameterSet,
    ParamValidationResult,
    PSFBlock,
    TwoStageResult,
)
from dlens.schemas._simulation import (
    CosmologyParams,
    SimConfig,
    SimModelConfig,
    SimOutput,
    SubstructureType,
)

__all__ = [
    # V1 simulation stage
    "SubstructureType",
    "SimModelConfig",
    "CosmologyParams",
    "SimConfig",
    "SimOutput",
    # V2 code generation
    "SimSpec",
    "ValidationResult",
    "CodegenResult",
    # Two-stage parameter extraction + validation
    "LensParameterSet",
    "InstrumentBlock",
    "PSFBlock",
    "ParamValidationResult",
    "CodeParamComparison",
    "FieldComparison",
    "TwoStageResult",
]
