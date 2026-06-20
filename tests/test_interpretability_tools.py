# tests/test_interpretability_tools.py
import torch
import torch.nn as nn
from dlens.tools.interpretability_tools import GradCAMTool, GradCAMInputSchema


class TinyConvClassifier(nn.Module):
    """A trivial conv classifier, just for testing Grad-CAM."""

    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv_layer1(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def test_gradcam_runs_and_returns_schema():
    model = TinyConvClassifier()
    tool = GradCAMTool(model=model)

    x = torch.randn(1, 1, 8, 8)  # single grayscale 8x8 "image"

    input_data = GradCAMInputSchema(
        input_tensor=x, target_class=0, target_layer_name="conv_layer1"
    )
    result = tool.arun(input_data)

    assert result.target_layer_name == "conv_layer1"
    assert len(result.heatmap_shape) == 4
    assert isinstance(result.heatmap_max_activation, float)
    assert isinstance(result.heatmap_mean_activation, float)