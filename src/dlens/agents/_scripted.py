# agents/_scripted.py
"""A scripted Pydantic AI ``FunctionModel`` for fully-offline demos and tests.

Lets the Data Simulation Agent run with NO GPU and NO live LLM: instead of calling
a real model it deterministically walks the canonical conversation —

    turn 1 (still ambiguous)  -> SimClarification
    turn 2+ (enough info)     -> run_simulation tool call (the proposed plan)
    after the tool has run     -> SimReport

Phase is detected structurally from the message history (number of user prompts;
presence of a ``run_simulation`` tool return), so it is robust to prompt text.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _output_tool(info: AgentInfo, fragment: str) -> str:
    """Find the auto-generated output tool whose name contains ``fragment``."""
    for tool in info.output_tools:
        if fragment in tool.name:
            return tool.name
    raise LookupError(
        f"No output tool matching {fragment!r}: {[t.name for t in info.output_tools]}"
    )


def make_scripted_sim_model(
    plan: dict[str, Any] | None = None,
    *,
    clarify_after: int = 1,
    question: str = "Which dark-matter substructure should I simulate?",
    options: tuple[str, ...] = ("no_sub", "cdm", "vortex"),
    report_message: str = "Simulation complete. Images, metadata.json and preview.png were written.",
) -> FunctionModel:
    """Build a scripted model that proposes ``plan`` once enough turns have passed.

    Args:
        plan: the ``SimConfig`` fields to propose (defaults to a CDM/Model_II run).
        clarify_after: ask a clarifying question while the user-prompt count is <= this.
    """
    plan = plan or {"substructure_type": "cdm", "model_config_name": "Model_II", "num_images": 4}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        n_user = sum(
            1 for m in messages for p in m.parts if getattr(p, "part_kind", "") == "user-prompt"
        )
        sim_ran = any(
            getattr(p, "part_kind", "") == "tool-return"
            and getattr(p, "tool_name", "") == "run_simulation"
            for m in messages
            for p in m.parts
        )

        if sim_ran:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        _output_tool(info, "SimReport"),
                        {"reasoning": "The approved simulation ran successfully.",
                         "message": report_message},
                    )
                ]
            )
        if n_user <= clarify_after:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        _output_tool(info, "SimClarification"),
                        {"reasoning": "The request lacks a substructure type; cannot default it.",
                         "question": question, "options": list(options)},
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("run_simulation", {"config": plan})])

    return FunctionModel(respond)
