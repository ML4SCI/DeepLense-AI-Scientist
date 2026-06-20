# src/dlens/tools/interpretability_tools.py
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from dlens.tools._base import AbstractBaseTool


class GradCAMInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_tensor: torch.Tensor       # shape: (1, C, H, W) - single image
    target_class: int
    target_layer_name: str           # e.g. "conv_layer3"


class GradCAMOutputSchema(BaseModel):
    target_class: int
    target_layer_name: str
    heatmap_shape: list[int]
    heatmap_max_activation: float
    heatmap_mean_activation: float


class GradCAMTool(AbstractBaseTool):
    """Deterministic tool: computes a Grad-CAM saliency heatmap for a
    given input image and target class, using a named convolutional layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations = None
        self._gradients = None
        super().__init__(input_schema=GradCAMInputSchema, output_schema=GradCAMOutputSchema)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def arun(self, input: GradCAMInputSchema) -> GradCAMOutputSchema:
        self.model.eval()

        target_layer = dict(self.model.named_modules())[input.target_layer_name]
        fwd_handle = target_layer.register_forward_hook(self._save_activation)
        bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

        try:
            x = input.input_tensor.clone().detach().requires_grad_(True)
            output = self.model(x)

            self.model.zero_grad()
            score = output[0, input.target_class]
            score.backward()

            # Global-average-pool the gradients -> per-channel weights
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self._activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)

            return GradCAMOutputSchema(
                target_class=input.target_class,
                target_layer_name=input.target_layer_name,
                heatmap_shape=list(cam.shape),
                heatmap_max_activation=float(cam.max().item()),
                heatmap_mean_activation=float(cam.mean().item()),
            )
        finally:
            fwd_handle.remove()
            bwd_handle.remove()