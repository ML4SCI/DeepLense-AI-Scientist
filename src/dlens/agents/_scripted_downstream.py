# agents/_scripted_downstream.py
"""Scripted Pydantic AI ``FunctionModel``s for offline train/infer/analysis tests.

Each deterministically calls its agent's tool with the given inputs, then echoes the
tool's result as the agent's (single) typed report — no GPU, no LLM, no network.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _tool_return(messages: list[ModelMessage], tool_name: str) -> Any:
    """Return the content of the most recent tool-return for ``tool_name``, or None."""
    found = None
    for m in messages:
        for p in m.parts:
            if getattr(p, "part_kind", "") == "tool-return" and getattr(p, "tool_name", "") == tool_name:
                found = p.content
    if isinstance(found, str):
        return json.loads(found)
    return found


def _report(info: AgentInfo, payload: dict) -> ModelResponse:
    # Each downstream agent has a single output type -> exactly one output tool.
    return ModelResponse(
        parts=[
            ToolCallPart(
                info.output_tools[0].name,
                {"reasoning": "Reporting the tool result.", **payload},
            )
        ]
    )


def make_scripted_train_model(
    dataset: dict[str, Any], architecture: dict[str, Any], training_config: dict[str, Any]
) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        rec = _tool_return(messages, "train_model")
        if rec is None:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "train_model",
                        {
                            "dataset": dataset,
                            "architecture": architecture,
                            "training_config": training_config,
                        },
                    )
                ]
            )
        return _report(info, {"message": f"Trained model (run {rec['run_id']}).", "result": rec})

    return FunctionModel(respond)


def make_scripted_infer_model(weights_path: str, dataset: dict[str, Any]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        rec = _tool_return(messages, "run_inference")
        if rec is None:
            return ModelResponse(
                parts=[ToolCallPart("run_inference", {"weights_path": weights_path, "dataset": dataset})]
            )
        # The tool now returns a compact summary; the report mirrors it.
        return _report(
            info,
            {
                "message": f"Scored {rec['num_samples']} samples.",
                "run_id": rec["run_id"],
                "num_samples": rec["num_samples"],
                "accuracy": rec.get("accuracy"),
            },
        )

    return FunctionModel(respond)


def make_scripted_analysis_model() -> FunctionModel:
    """The analysis tool takes no arguments — the InferResult rides on deps."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        rec = _tool_return(messages, "analyze_predictions")
        if rec is None:
            return ModelResponse(parts=[ToolCallPart("analyze_predictions", {})])
        return _report(info, {"message": rec["summary"], "result": rec})

    return FunctionModel(respond)
