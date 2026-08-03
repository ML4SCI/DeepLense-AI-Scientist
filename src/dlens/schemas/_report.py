# schemas/_report.py
"""Schema for the final report produced at the end of an experiment campaign."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from dlens.agents._base import OutputSchema


class ReportDocument(OutputSchema):
    """The Report Agent's structured output (agent #7)."""

    title: str = Field(description="Short title for the report.")
    hypothesis: str = Field(description="The original hypothesis / problem statement.")
    summary: str = Field(description="What was tried across iterations and how it evolved.")
    best_accuracy: Optional[float] = Field(
        default=None, description="Best accuracy achieved across iterations."
    )
    final_architecture: Optional[str] = Field(
        default=None, description="Architecture that produced the best result."
    )
    findings: list[str] = Field(
        default_factory=list, description="Key factual findings, one per item."
    )
    conclusion: str = Field(description="Short factual conclusion.")
