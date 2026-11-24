"""Training objectives for reconstruction, adversarial, and diffusion SR."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dlens.sr_advanced_models import PatchDiscriminator
from dlens.sr_losses import (
    COMPOSITE_WEIGHTS,
    default_loss,
    reconstruction_loss,
    validate_loss,
)
from dlens.sr_models import SRArchitecture


class SRTrainer:
    def __init__(self, model: nn.Module, spec: SRArchitecture, learning_rate: float,
                 device: torch.device, loss_id: str | None = None):
        self.model = model
        self.spec = spec
        self.loss_id = loss_id or default_loss(spec.family)
        validate_loss(spec.family, self.loss_id)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.discriminator = None
        self.discriminator_optimizer = None
        if spec.family == "realesrgan":
            self.discriminator = PatchDiscriminator(spec.width).to(device)
            self.discriminator_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(), lr=learning_rate)

    @property
    def objective(self) -> str:
        if self.loss_id == "l1_adversarial":
            return "reconstruction_plus_adversarial"
        return ("conditional_" if self.spec.family == "sr3" else "reconstruction_") + self.loss_id

    def step(self, lr: torch.Tensor, hr: torch.Tensor, l1_weight: float) -> float:
        self.model.train()
        if self.spec.family == "sr3":
            loss = self.model.training_loss(lr, hr, self.loss_id)
        else:
            prediction = self.model(lr)
            reconstruction = reconstruction_loss(prediction, hr, self.loss_id, l1_weight)
            if self.discriminator is None:
                loss = reconstruction
            else:
                self.discriminator.train()
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = self.discriminator(hr)
                fake_logits = self.discriminator(prediction.detach())
                d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                          + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))) / 2
                d_loss.backward()
                self.discriminator_optimizer.step()
                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(False)
                fake_logits = self.discriminator(prediction)
                adversarial = F.binary_cross_entropy_with_logits(fake_logits,
                                                                   torch.ones_like(fake_logits))
                loss = reconstruction + COMPOSITE_WEIGHTS[self.loss_id] * adversarial
                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    @property
    def loss_config(self) -> dict:
        return {"loss_id": self.loss_id,
                "composite_weight": COMPOSITE_WEIGHTS.get(self.loss_id)}

    def checkpoint_extra(self) -> dict:
        return ({"discriminator": self.discriminator.cpu().state_dict()}
                if self.discriminator is not None else {})

    def training_state(self) -> dict:
        """Save optimizer state so a later stage truly continues training."""
        state = {"optimizer": self.optimizer.state_dict()}
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
            state["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return state

    def load_training_state(self, checkpoint: dict) -> None:
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise ValueError("checkpoint lacks optimizer state required for continuation")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

    def set_learning_rate(self, learning_rate: float) -> None:
        for optimizer in (self.optimizer, self.discriminator_optimizer):
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
