# agents/_scripted_search.py
"""Scripted FunctionModels for offline architecture-search tests (no LLM)."""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


def make_scripted_generator(batches: list[list[dict[str, Any]]]) -> FunctionModel:
    """Emit one candidate batch per call (first call -> batches[0], etc.)."""
    state = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        batch = batches[min(state["n"], len(batches) - 1)]
        state["n"] += 1
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name,
            {"reasoning": "scripted candidates", "candidates": batch},
        )])

    return FunctionModel(respond)


def make_scripted_judge(ranking: list[int]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name,
            {"reasoning": "scripted ranking", "ranking": ranking},
        )])

    return FunctionModel(respond)
