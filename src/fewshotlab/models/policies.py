"""Which parameters each method may train, and which modules stay in eval mode.

The policy is explicit about both ``requires_grad`` and module modes, because ``requires_grad``
alone does not stop BatchNorm from updating its running statistics in train mode. Call
:func:`enter_train_mode` instead of ``model.train()``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from fewshotlab.config import OptimizerSettings
from fewshotlab.errors import ConfigError
from fewshotlab.models.backbone import ResNet18Classifier

METHODS = ("scratch", "frozen", "partial", "prototypes")


def _check(method: str) -> None:
    if method not in METHODS:
        raise ConfigError(f"unknown method {method!r}; choose from {list(METHODS)}")


def is_trainable_name(method: str, name: str) -> bool:
    """Whether parameter ``name`` (from ``named_parameters()``) may train under ``method``."""
    _check(method)
    if method == "scratch":
        return True
    if method == "frozen":
        return name.startswith("head.")
    if method == "partial":
        return name.startswith(("head.", "encoder.layer4."))
    return False  # prototypes: nothing trains


def apply_trainable(model: ResNet18Classifier, method: str) -> list[str]:
    """Set ``requires_grad`` per the policy; return the trainable parameter names."""
    trainable = []
    for name, parameter in model.named_parameters():
        allowed = is_trainable_name(method, name)
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(name)
    return trainable


def _batchnorms(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def enter_train_mode(model: ResNet18Classifier, method: str) -> None:
    """``model.train()`` followed by the method's frozen-module policy."""
    _check(method)
    model.train()
    if method == "frozen":
        model.encoder.eval()  # frozen encoder: no BatchNorm statistic updates
    elif method == "partial":
        for bn in _batchnorms(model):
            bn.eval()  # every running mean/variance stays fixed; layer4 affine params still train
    elif method == "prototypes":
        model.eval()


def enter_eval_mode(model: ResNet18Classifier) -> None:
    model.eval()


def check_train_mode(model: ResNet18Classifier, method: str) -> None:
    """Raise if module modes or ``requires_grad`` flags disagree with the policy."""
    problems = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != is_trainable_name(method, name):
            problems.append(f"{name}: requires_grad={parameter.requires_grad}")
    for bn in _batchnorms(model):
        should_train = method == "scratch"
        if bn.training != should_train:
            problems.append(f"BatchNorm {bn} training={bn.training}")
    if problems:
        raise RuntimeError(f"{method} policy violated: {problems[:5]}")


def optimizer_groups(
    model: ResNet18Classifier, method: str, settings: OptimizerSettings
) -> list[dict[str, Any]]:
    """AdamW parameter groups. Every group uses the configured weight decay."""
    _check(method)
    named = dict(model.named_parameters())
    head = [p for n, p in named.items() if n.startswith("head.")]
    if method == "scratch":
        return [{"name": "all", "params": list(named.values()), "lr": settings.lr_scratch}]
    if method == "frozen":
        return [{"name": "head", "params": head, "lr": settings.lr_head}]
    if method == "partial":
        layer4 = [p for n, p in named.items() if n.startswith("encoder.layer4.")]
        return [
            {"name": "layer4", "params": layer4, "lr": settings.lr_layer4},
            {"name": "head", "params": head, "lr": settings.lr_head},
        ]
    return []  # prototypes: no optimizer
