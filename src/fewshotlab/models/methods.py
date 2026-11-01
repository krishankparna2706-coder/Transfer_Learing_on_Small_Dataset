"""Method registry and the shared prediction path used by the trainer, evaluator and app."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fewshotlab.errors import ConfigError
from fewshotlab.models import prototypes as proto
from fewshotlab.models.backbone import ResNet18Classifier, WeightsInfo, build_resnet18
from fewshotlab.models.policies import apply_trainable, enter_eval_mode


@dataclass(frozen=True)
class MethodInfo:
    name: str
    init: str  # "random" (scratch) or "imagenet" (transfer methods)
    learned: bool  # trained by gradient descent with a checkpoint
    score_kind: str


REGISTRY: dict[str, MethodInfo] = {
    "scratch": MethodInfo("scratch", "random", True, "softmax(logits); uncalibrated model score"),
    "frozen": MethodInfo("frozen", "imagenet", True, "softmax(logits); uncalibrated model score"),
    "partial": MethodInfo("partial", "imagenet", True, "softmax(logits); uncalibrated model score"),
    "prototypes": MethodInfo(
        "prototypes",
        "imagenet",
        False,
        "softmax(-squared Euclidean distance) at scale 1; uncalibrated model score",
    ),
}


def get_method(name: str) -> MethodInfo:
    if name not in REGISTRY:
        raise ConfigError(f"unknown method {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def build_model(
    method: str,
    num_classes: int,
    *,
    init_seed: int,
    weights_dir: Path | None = None,
    allow_download: bool = False,
    init_override: str | None = None,
) -> tuple[ResNet18Classifier, WeightsInfo]:
    """A fresh model for one run with the method's parameter policy applied.

    ``init_override="random"`` exists only for software fixtures that must run without the
    pretrained weights; protocol runs never use it.
    """
    info = get_method(method)
    model, weights = build_resnet18(
        num_classes,
        init=init_override or info.init,
        init_seed=init_seed,
        weights_dir=weights_dir,
        allow_download=allow_download,
    )
    apply_trainable(model, method)
    return model, weights


@dataclass
class Outputs:
    """Predictions in loader order. ``raw`` is logits, or negative squared distances."""

    indices: np.ndarray
    labels: np.ndarray
    raw: np.ndarray
    scores: np.ndarray
    pred: np.ndarray


@torch.inference_mode()
def collect_logits(model: ResNet18Classifier, loader: DataLoader, device: str) -> Outputs:
    enter_eval_mode(model)
    raws, labels, indices = [], [], []
    for x, y, index in loader:
        logits = model(x.to(device)).float().to("cpu")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite logits during evaluation")
        raws.append(logits)
        labels.append(torch.as_tensor(y))
        indices.append(torch.as_tensor(index))
    raw = torch.cat(raws)
    return Outputs(
        indices=torch.cat(indices).numpy(),
        labels=torch.cat(labels).numpy(),
        raw=raw.numpy(),
        scores=torch.softmax(raw, dim=1).numpy(),
        pred=torch.argmax(raw, dim=1).numpy(),
    )


@torch.inference_mode()
def collect_embeddings(
    model: ResNet18Classifier, loader: DataLoader, device: str
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    enter_eval_mode(model)
    feats, labels, indices = [], [], []
    for x, y, index in loader:
        feats.append(model.features(x.to(device)).float().to("cpu"))
        labels.append(torch.as_tensor(y))
        indices.append(torch.as_tensor(index))
    return torch.cat(feats), torch.cat(labels).numpy(), torch.cat(indices).numpy()


def prototype_outputs(
    prototypes: torch.Tensor,
    query_embeddings: torch.Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    ids: Sequence[str] | None = None,
) -> Outputs:
    normalized = proto.normalize_embeddings(query_embeddings, ids)
    distances = proto.squared_distances(normalized, prototypes)
    return Outputs(
        indices=np.asarray(indices),
        labels=np.asarray(labels),
        raw=(-distances).numpy(),
        scores=proto.scores(distances).numpy(),
        pred=proto.predict(distances).numpy(),
    )
