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


def build_model(arch: ArchitectureSpec, dropout: float = 0.0):
    """Build a real torch model from the spec (only BUILDABLE_FAMILIES).

    * ``resnet`` — BasicBlock ResNet; stages from ``depths``/``widths`` (falls back
      to the named preset, e.g. resnet18/resnet34, with classic widths).
    * ``cnn``    — VGG-like plain convnet: ``depths[i]`` 3x3 convs at ``widths[i]``
      channels per stage, maxpool between stages.
    * ``vit``, ``mlpmixer``, ``hybrid``, ``equivariant`` — delegated to
      ``dlens.tools._arch_families``; see that module for how depths/widths map.

    A Dropout layer is always present before the classifier (p=0 disables it) so
    checkpoint state_dict indices are stable regardless of the dropout setting.
    """
    _require_torch()
    from torch import nn

    from dlens.schemas._model_design import ArchFamily

    if not arch.is_buildable():
        raise ValueError(f"Architecture family '{arch.family.value}' is not buildable.")

    num_classes = arch.num_classes or 2
    in_ch = arch.channels or 1
    if arch.depths is not None:
        blocks, widths = tuple(arch.depths), tuple(arch.widths)
    else:
        blocks = _BLOCKS_BY_NAME.get(arch.name.lower(), (2, 2, 2, 2))
        widths = (64, 128, 256, 512)[: len(blocks)]

    # Families beyond cnn/resnet live in _arch_families (vit, mlpmixer, hybrid,
    # equivariant). cnn/resnet stay here untouched: the published runs depend on
    # their exact behaviour.
    if arch.family not in (ArchFamily.CNN, ArchFamily.RESNET):
        from dlens.tools._arch_families import build_family

        return build_family(arch, dropout=dropout)

    if arch.family == ArchFamily.CNN:
        layers: list = []
        cin = in_ch
        for d, w in zip(blocks, widths):
            for _ in range(d):
                layers += [nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w),
                           nn.ReLU(inplace=True)]
                cin = w
            layers.append(nn.MaxPool2d(2))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                   nn.Linear(widths[-1], num_classes)]
        return nn.Sequential(*layers)

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

    stem = [
        nn.Conv2d(in_ch, widths[0], 7, 2, 3, bias=False),
        nn.BatchNorm2d(widths[0]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
    ]
    stages = []
    cin = widths[0]
    for s, (d, w) in enumerate(zip(blocks, widths)):
        stages.append(stage(cin, w, d, 1 if s == 0 else 2))
        cin = w
    head = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes)]
    return nn.Sequential(*stem, *stages, *head)


# Backwards-compatible alias (train/infer call sites predate the cnn family).
_build_resnet = build_model


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

    def __init__(
        self, output_root: str = "models", device: str | None = None, seed: int = 0
    ) -> None:
        self._output_root = output_root
        self._device_override = device
        # Training seed. Default 0 reproduces every single-seed run on this branch;
        # vary it for replication, where the spread across seeds IS the
        # measurement. Mirrors exp/multi-seed exactly so results are comparable
        # with the multi-seed pilot's resnet34 figures.
        self._seed = seed

    @property
    def seed(self) -> int:
        """The training seed this backend was constructed with."""
        return self._seed

    def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, config: TrainingConfig
    ) -> TrainResult:
        torch = _require_torch()
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self._device_override) if self._device_override else _device(torch)
        # Seeded so iterations differ only by the config changes being tested,
        # not by initialization/shuffling noise.
        torch.manual_seed(self._seed)
        x, y = _load_images(dataset)
        mean, std = float(x.mean()), float(x.std() or 1.0)
        x = (x - mean) / std

        # Early stopping: hold out a seeded 10% slice of TRAIN (never touches val/).
        es_x = es_y = None
        if config.early_stop_patience > 0:
            rng = np.random.default_rng(self._seed)
            idx = rng.permutation(len(y))
            n_hold = max(1, len(y) // 10)
            hold, keep = idx[:n_hold], idx[n_hold:]
            es_x, es_y = x[hold], y[hold]
            x, y = x[keep], y[keep]

        model = _build_resnet(architecture, dropout=config.dropout).to(device)
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

        def _augment(xb):
            # Lensing images tolerate flips/90-degree rotations (orientation symmetry).
            if int(torch.randint(0, 2, (1,))) == 1:
                xb = torch.flip(xb, dims=[3])
            if int(torch.randint(0, 2, (1,))) == 1:
                xb = torch.flip(xb, dims=[2])
            return torch.rot90(xb, k=int(torch.randint(0, 4, (1,))), dims=[2, 3])

        def _holdout_acc() -> float:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(es_x), 256):
                    xb = torch.from_numpy(es_x[i : i + 256]).to(device)
                    yb = torch.from_numpy(es_y[i : i + 256]).to(device)
                    correct += int((model(xb).argmax(1) == yb).sum())
            model.train()
            return correct / len(es_y)

        model.train()
        final_loss, final_acc = 0.0, 0.0
        best_acc, best_state, since_best = -1.0, None, 0
        epochs_run = 0
        for epoch in range(config.epochs):
            tot, correct, loss_sum = 0, 0, 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if config.augment:
                    xb = _augment(xb)
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
            epochs_run = epoch + 1
            line = f"    epoch {epochs_run}/{config.epochs}  loss={final_loss:.4f}  acc={final_acc:.4f}"
            if es_x is not None:
                hold_acc = _holdout_acc()
                line += f"  holdout={hold_acc:.4f}"
                if hold_acc > best_acc + 1e-4:
                    best_acc, since_best = hold_acc, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    since_best += 1
            print(line, flush=True)
            if es_x is not None and since_best >= config.early_stop_patience:
                print(f"    early stop at epoch {epochs_run} (best holdout={best_acc:.4f})", flush=True)
                break
        if best_state is not None:
            model.load_state_dict(best_state)

        run_id = str(uuid.uuid4())[:8]
        out_dir = Path(self._output_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(out_dir / "model.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": architecture.model_dump(mode="json"),
                "dropout": config.dropout,
                "mean": mean, "std": std,
                "class_names": dataset.class_names,
            },
            weights_path,
        )
        metrics = {"train_accuracy": round(final_acc, 4), "final_loss": round(final_loss, 4)}
        if best_acc >= 0:
            metrics["holdout_accuracy"] = round(best_acc, 4)
        return TrainResult(
            run_id=run_id,
            weights_path=weights_path,
            backend=self.name,
            num_classes=dataset.num_classes,
            metrics=metrics,
            epochs_run=epochs_run,
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
        model = _build_resnet(arch, dropout=ckpt.get("dropout", 0.0)).to(device)
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
