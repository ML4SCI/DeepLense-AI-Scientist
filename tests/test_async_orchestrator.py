# tests/test_async_orchestrator.py
import pytest
import torch
import torch.nn as nn
from dlens.tools.robustness_tools import FGSMAttackTool, FGSMInputSchema
from dlens.tools.interpretability_tools import GradCAMTool, GradCAMInputSchema
from dlens.agents.async_orchestrator import AsyncAgentOrchestrator, AsyncOrchestratorInput, ToolRun


class TinyConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv_layer1(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


@pytest.mark.asyncio
async def test_concurrent_evaluation_with_heterogeneous_tools():
    """Real test: FGSM and GradCAM together, each gets correctly-shaped input."""
    model = TinyConvClassifier()
    fgsm_tool = FGSMAttackTool(model=model)
    gradcam_tool = GradCAMTool(model=model)

    x = torch.randn(1, 1, 8, 8)
    y = torch.tensor([0])

    orchestrator = AsyncAgentOrchestrator()
    input_data = AsyncOrchestratorInput(
        tool_runs=[
            ToolRun(
                tool=fgsm_tool,
                input_builder=lambda: FGSMInputSchema(input_tensor=x, labels=y, epsilon=0.1),
            ),
            ToolRun(
                tool=gradcam_tool,
                input_builder=lambda: GradCAMInputSchema(
                    input_tensor=x, target_class=0, target_layer_name="conv_layer1"
                ),
            ),
        ]
    )

    report = await orchestrator.execute_concurrent_evaluation(input_data)

    assert report.tool_count == 2
    assert report.all_tools_succeeded is True
    assert report.errors == {}
    assert "FGSMAttackTool" in report.metrics
    assert "GradCAMTool" in report.metrics


@pytest.mark.asyncio
async def test_orchestrator_reports_partial_failure_without_crashing():
    """If one tool's builder raises (e.g. bad input), the orchestrator
    should report it in `errors`, not silently claim success."""
    model = TinyConvClassifier()
    fgsm_tool = FGSMAttackTool(model=model)

    def broken_builder():
        raise ValueError("simulated bad input")

    orchestrator = AsyncAgentOrchestrator()
    input_data = AsyncOrchestratorInput(
        tool_runs=[ToolRun(tool=fgsm_tool, input_builder=broken_builder)]
    )

    report = await orchestrator.execute_concurrent_evaluation(input_data)

    assert report.all_tools_succeeded is False
    assert "FGSMAttackTool" in report.errors
    assert "simulated bad input" in report.errors["FGSMAttackTool"]