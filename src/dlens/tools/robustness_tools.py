# src/dlens/tools/robustness_tools.py
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from dlens.tools._base import AbstractBaseTool


class FGSMInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_tensor: torch.Tensor
    labels: torch.Tensor
    epsilon: float = 0.1


class FGSMOutputSchema(BaseModel):
    attack_type: str = "FGSM"
    epsilon: float
    prediction_changed: bool
    loss_before_attack: float


class FGSMAttackTool(AbstractBaseTool):
    """Deterministic tool: runs a single FGSM perturbation against a
    PyTorch classifier and reports whether the prediction flips."""

    def __init__(self, model: nn.Module):
        self.model = model
        super().__init__(input_schema=FGSMInputSchema, output_schema=FGSMOutputSchema)

    def arun(self, input: FGSMInputSchema) -> FGSMOutputSchema:
        self.model.eval()
        x = input.input_tensor.clone().detach().requires_grad_(True)

        output = self.model(x)
        loss = nn.functional.cross_entropy(output, input.labels)
        self.model.zero_grad()
        loss.backward()

        perturbed = torch.clamp(x + input.epsilon * x.grad.sign(), 0, 1)
        with torch.no_grad():
            perturbed_output = self.model(perturbed)

        changed = bool((output.argmax(dim=1) != perturbed_output.argmax(dim=1)).any().item())
        return FGSMOutputSchema(
            epsilon=input.epsilon,
            prediction_changed=changed,
            loss_before_attack=float(loss.item()),
        )