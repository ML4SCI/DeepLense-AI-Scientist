# tools/_torch_backends.py
"""Real (torch) training and inference backends for the Train / Infer agents.

Builds a genuine CNN from the Model Design agent's ``ArchitectureSpec`` (a
from-scratch ResNet — no torchvision dependency, single-channel friendly), trains
it with the ``TrainingConfig`` hyperparameters on MPS (Apple Silicon) or CPU, and
runs held-out inference from the saved checkpoint.

torch is imported lazily so the framework (and the offline test suite) works
without it; install with the ``training`` extra.
"""

from __future__ import annotations

import glob
import os
import uuid
from pathlib import Path

import numpy as np

from dlens.schemas._downstream import DatasetRef, InferResult, TrainResult
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig

_BLOCKS_BY_NAME = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}


def _require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "The torch training backend requires PyTorch: uv pip install torch "
            "(or install the 'training' extra)."
        ) from exc
    import torch

    return torch


def _device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_resnet(arch: ArchitectureSpec):
    """A compact ResNet built from the spec (name -> depth; channels; classes)."""
    _require_torch()
    from torch import nn

    blocks = _BLOCKS_BY_NAME.get(arch.name.lower(), (2, 2, 2, 2))
    num_classes = arch.num_classes or 2
    in_ch = arch.channels or 1

    class BasicBlock(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)
            self.down = None
            if stride != 1 or cin != cout:
                self.down = nn.Sequential(
                    nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
                )

        def forward(self, x):
            idn = x if self.down is None else self.down(x)
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.act(out + idn)

    def stage(cin, cout, n, stride):
        layers = [BasicBlock(cin, cout, stride)]
        layers += [BasicBlock(cout, cout) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    widths = (64, 128, 256, 512)
    return nn.Sequential(
        nn.Conv2d(in_ch, widths[0], 7, 2, 3, bias=False),
        nn.BatchNorm2d(widths[0]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
        stage(widths[0], widths[0], blocks[0], 1),
        stage(widths[0], widths[1], blocks[1], 2),
        stage(widths[1], widths[2], blocks[2], 2),
        stage(widths[2], widths[3], blocks[3], 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(widths[3], num_classes),
    )


def _load_images(ref: DatasetRef) -> tuple[np.ndarray, np.ndarray]:
    """Load ``<class>_*.npy`` under ``ref.root`` into (X[N,1,H,W] float32, y[N])."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, name in enumerate(ref.class_names):
        for fp in sorted(glob.glob(os.path.join(ref.root, f"{name}_*.npy"))):
            item = np.load(fp, allow_pickle=True)
            if item.dtype == object:  # official axion files store [image, axion_mass]
                item = np.asarray(item[0])
            xs.append(item.astype(np.float32))
            ys.append(label)
    if not xs:
        raise RuntimeError(f"No .npy samples under {ref.root!r} for {ref.class_names}.")
    x = np.stack(xs)[:, None, :, :]
    return x, np.asarray(ys, dtype=np.int64)


class TorchTrainBackend:
    """Real CNN training driven by ArchitectureSpec + TrainingConfig."""

    name = "torch"

    def __init__(self, output_root: str = "models", device: str | None = None) -> None:
        self._output_root = output_root
        self._device_override = device

    def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, config: TrainingConfig
    ) -> TrainResult:
        torch = _require_torch()
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self._device_override) if self._device_override else _device(torch)
        x, y = _load_images(dataset)
        mean, std = float(x.mean()), float(x.std() or 1.0)
        x = (x - mean) / std

        model = _build_resnet(architecture).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=config.batch_size, shuffle=True,
        )
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
            if config.lr_scheduler == "cosine" else None
        )
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        final_loss, final_acc = 0.0, 0.0
        for epoch in range(config.epochs):
            tot, correct, loss_sum = 0, 0, 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()) * len(yb)
                correct += int((logits.argmax(1) == yb).sum())
                tot += len(yb)
            if sched is not None:
                sched.step()
            final_loss, final_acc = loss_sum / tot, correct / tot
            print(f"    epoch {epoch + 1}/{config.epochs}  loss={final_loss:.4f}  acc={final_acc:.4f}", flush=True)

        run_id = str(uuid.uuid4())[:8]
        out_dir = Path(self._output_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(out_dir / "model.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": architecture.model_dump(mode="json"),
                "mean": mean, "std": std,
                "class_names": dataset.class_names,
            },
            weights_path,
        )
        return TrainResult(
            run_id=run_id,
            weights_path=weights_path,
            backend=self.name,
            num_classes=dataset.num_classes,
            metrics={"train_accuracy": round(final_acc, 4), "final_loss": round(final_loss, 4)},
            epochs_run=config.epochs,
        )


class TorchInferBackend:
    """Inference from a TorchTrainBackend checkpoint."""

    name = "torch"

    def __init__(self, device: str | None = None) -> None:
        self._device_override = device

    def infer(self, weights_path: str, dataset: DatasetRef) -> InferResult:
        torch = _require_torch()

        device = torch.device(self._device_override) if self._device_override else _device(torch)
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        arch = ArchitectureSpec(**ckpt["architecture"])
        model = _build_resnet(arch).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        x, y = _load_images(dataset)
        x = (x - ckpt["mean"]) / ckpt["std"]
        probs_all: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                xb = torch.from_numpy(x[i : i + 256]).to(device)
                probs_all.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        probs = np.concatenate(probs_all)
        preds = probs.argmax(1)
        return InferResult(
            run_id=str(uuid.uuid4())[:8],
            num_samples=int(len(y)),
            num_classes=int(probs.shape[1]),
            predictions=preds.astype(int).tolist(),
            probabilities=probs.tolist(),
            true_labels=y.astype(int).tolist(),
            accuracy=float((preds == y).mean()),
            backend=self.name,
        )
