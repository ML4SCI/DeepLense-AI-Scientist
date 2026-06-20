# tests/test_robustness_tools.py
import torch
import torch.nn as nn
from dlens.tools.robustness_tools import FGSMAttackTool, FGSMInputSchema


class TinyClassifier(nn.Module):
    """A trivial 2-class linear classifier, just for testing the attack tool."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_fgsm_attack_runs_and_returns_schema():
    model = TinyClassifier()
    tool = FGSMAttackTool(model=model)

    x = torch.randn(3, 4)          # batch of 3 samples, 4 features each
    y = torch.tensor([0, 1, 0])    # dummy labels

    input_data = FGSMInputSchema(input_tensor=x, labels=y, epsilon=0.1)
    result = tool.arun(input_data)

    assert result.attack_type == "FGSM"
    assert result.epsilon == 0.1
    assert isinstance(result.prediction_changed, bool)
    assert isinstance(result.loss_before_attack, float)
    assert result.loss_before_attack >= 0