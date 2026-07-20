# agents/_scripted_planner.py
"""Scripted ``FunctionModel``s for offline planner / report tests.

The planner model reads the serialized ``ExperimentState`` from the prompt and
decides based on ``current_iteration`` (deterministic, no LLM). The report model
summarizes the same state. Both let the full feedback loop run with no LLM/network.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _user_json(messages: list[ModelMessage]) -> dict:
    """Parse the latest user-prompt (the serialized ExperimentState) into a dict."""
    text: Optional[str] = None
    for m in messages:
        for p in m.parts:
            if getattr(p, "part_kind", "") == "user-prompt":
                text = p.content
    try:
        return json.loads(text) if text else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def make_scripted_planner_model(
    *,
    report_after: int = 1,
    terminal_action: str = "report",
    updated_params: Optional[dict[str, Any]] = None,
) -> FunctionModel:
    """Continue (``design_model``) until ``current_iteration >= report_after``, then
    emit ``terminal_action`` (``report`` or ``stop``)."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        iteration = _user_json(messages).get("current_iteration", 0)
        if iteration >= report_after:
            decision = {"action": terminal_action, "rationale": "Budget/metric reached.", "updated_params": {}}
        else:
            decision = {
                "action": "design_model",
                "rationale": "Adjust configuration and try again.",
                "updated_params": updated_params or {"learning_rate": 0.0001},
            }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, decision)])

    return FunctionModel(respond)


def make_scripted_report_model() -> FunctionModel:
    """Summarize the experiment state into a ReportDocument (best accuracy + arch)."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state = _user_json(messages)
        runs = state.get("runs", [])
        best_acc: Optional[float] = None
        best_arch: Optional[str] = None
        for r in runs:
            acc = (r.get("analysis_result") or {}).get("accuracy")
            if acc is not None and (best_acc is None or acc > best_acc):
                best_acc = acc
                best_arch = (r.get("architecture") or {}).get("name")
        doc = {
            "reasoning": "Summarized the experiment campaign from its history.",
            "title": "Experiment report",
            "hypothesis": state.get("hypothesis", ""),
            "summary": f"Ran {len(runs)} iteration(s); best accuracy {best_acc}.",
            "best_accuracy": best_acc,
            "final_architecture": best_arch,
            "findings": [f"{len(runs)} iteration(s) completed."],
            "conclusion": "See per-iteration metrics for details.",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, doc)])

    return FunctionModel(respond)
