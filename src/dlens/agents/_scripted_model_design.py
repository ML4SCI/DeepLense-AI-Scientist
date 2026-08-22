# agents/_scripted_model_design.py
"""Scripted Pydantic AI ``FunctionModel`` for offline Model Design Agent tests/demos.

Deterministically drives the agent with NO GPU and NO live LLM: it calls
``recommend_architecture`` with the given dataset characteristics, then echoes the
returned recommendation as a ``ModelDesignReport``. ``ModelDesignReport`` is the
agent's single output type, so the output tool is taken directly (the lone entry in
``info.output_tools``).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


def make_scripted_model_design_model(characteristics: dict[str, Any]) -> FunctionModel:
    """Call ``recommend_architecture`` with ``characteristics``, then report it back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        rec = None
        for m in messages:
            for p in m.parts:
                if (
                    getattr(p, "part_kind", "") == "tool-return"
                    and getattr(p, "tool_name", "") == "recommend_architecture"
                ):
                    rec = p.content

        if rec is None:
            return ModelResponse(
                parts=[ToolCallPart("recommend_architecture", {"characteristics": characteristics})]
            )

        if isinstance(rec, str):
            import json

            rec = json.loads(rec)
        # ModelDesignReport is the single output type -> exactly one output tool.
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "reasoning": "Adopting the vetted baseline from recommend_architecture.",
                        "message": f"Recommended {rec['architecture']['name']}.",
                        "architecture": rec["architecture"],
                        "training_config": rec["training_config"],
                    },
                )
            ]
        )

    return FunctionModel(respond)
