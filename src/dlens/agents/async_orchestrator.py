# src/dlens/agents/async_orchestrator.py
import asyncio
from typing import Any, Callable, Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from dlens.agents._base import AbstractBaseAgent
from dlens.tools._base import AbstractBaseTool


class ToolRun(BaseModel):
    """Pairs a tool with a zero-arg builder that constructs ITS OWN
    input schema instance. This avoids assuming every tool shares the
    same field names (e.g. FGSM needs `labels`+`epsilon`, GradCAM needs
    `target_class`+`target_layer_name` — they are not interchangeable)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool: AbstractBaseTool
    input_builder: Callable[[], BaseModel]


class AsyncOrchestratorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_runs: List[ToolRun] = Field(
        description="Heterogeneous tools to run concurrently, each with its own input builder."
    )


class AsyncOrchestratorOutput(BaseModel):
    orchestration_mode: str
    metrics: Dict[str, Any] = Field(description="Per-tool output, keyed by tool class name.")
    tool_count: int
    all_tools_succeeded: bool
    errors: Dict[str, str] = Field(default_factory=dict)


class AsyncAgentOrchestrator(AbstractBaseAgent):
    """
    Runs multiple heterogeneous deterministic tools concurrently.

    Note: tool.arun() is synchronous, CPU-bound PyTorch code. Wrapping
    it in `async def` alone does NOT parallelize it — asyncio.to_thread()
    is required to actually run it on a separate thread.
    """

    def __init__(self):
        super().__init__(input_schema=AsyncOrchestratorInput, output_schema=AsyncOrchestratorOutput)

    async def _run_one(self, run: ToolRun) -> Tuple[str, Dict[str, Any], bool, str]:
        name = run.tool.__class__.__name__
        try:
            tool_input = run.input_builder()
            result = await asyncio.to_thread(run.tool.arun, tool_input)
            return name, result.model_dump(), True, ""
        except Exception as e:
            return name, {}, False, str(e)

    async def execute_concurrent_evaluation(self, input: AsyncOrchestratorInput) -> AsyncOrchestratorOutput:
        results = await asyncio.gather(*(self._run_one(r) for r in input.tool_runs))

        metrics, errors, all_ok = {}, {}, True
        for name, payload, ok, err in results:
            if ok:
                metrics[name] = payload
            else:
                errors[name] = err
                all_ok = False

        return AsyncOrchestratorOutput(
            orchestration_mode="asyncio.gather + to_thread (real thread-pool concurrency)",
            metrics=metrics,
            tool_count=len(results),
            all_tools_succeeded=all_ok,
            errors=errors,
        )