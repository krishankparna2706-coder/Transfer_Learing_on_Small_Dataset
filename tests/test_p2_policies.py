"""P2: each method changes exactly the parameters it may train; frozen BatchNorm stays frozen.

The policy tests use randomly initialized weights as a stand-in for ImageNet weights: the policy
does not depend on the starting values. ``test_p2_real_weights.py`` repeats the check with the
real IMAGENET1K_V1 weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fewshotlab.config import OptimizerSettings
from fewshotlab.models.methods import build_model
from fewshotlab.models.policies import (
    check_train_mode,
    enter_train_mode,
    is_trainable_name,
    optimizer_groups,
)


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(4, 3, 64, 64, generator=generator), torch.tensor([0, 1, 2, 0])


def snapshot(model):
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    return params, buffers


def one_step(model, method, x, y):
    groups = optimizer_groups(model, method, OptimizerSettings())
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    enter_train_mode(model, method)
    check_train_mode(model, method)
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    return groups, finite


def assert_policy_after_one_step(model, method, batch):
    before_params, before_buffers = snapshot(model)
    groups, finite = one_step(model, method, *batch)
    assert finite
    after_params, after_buffers = snapshot(model)
    changed = {n for n in before_params if not torch.equal(before_params[n], after_params[n])}
    allowed = {n for n in before_params if is_trainable_name(method, n)}
    assert changed == allowed, sorted(changed ^ allowed)[:5]
    in_optimizer = {id(p) for g in groups for p in g["params"]}
    assert in_optimizer == {id(p) for n, p in model.named_parameters() if n in allowed}
    changed_buffers = {
        n for n in before_buffers if not torch.equal(before_buffers[n], after_buffers[n])
    }
    if method == "scratch":
        assert any(n.endswith("running_mean") for n in changed_buffers)
    else:
        assert changed_buffers == set()  # every running statistic stays fixed


@pytest.mark.parametrize("method", ["scratch", "frozen", "partial"])
def test_only_permitted_parameters_and_buffers_change(method, batch):
    model, _ = build_model(method, 3, init_seed=0, init_override="random")
    assert_policy_after_one_step(model, method, batch)


def test_partial_trains_layer4_affine_but_freezes_every_running_statistic():
    model, _ = build_model("partial", 3, init_seed=0, init_override="random")
    named = dict(model.named_parameters())
    assert named["encoder.layer4.0.bn1.weight"].requires_grad
    assert not named["encoder.layer3.0.bn1.weight"].requires_grad
    assert not named["encoder.conv1.weight"].requires_grad
    enter_train_mode(model, "partial")
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(bn.training for bn in bns)


def test_plain_train_call_would_update_frozen_statistics(batch):
    """Negative control: without the policy, model.train() changes BatchNorm buffers."""
    model, _ = build_model("frozen", 3, init_seed=0, init_override="random")
    _, before = snapshot(model)
    model.train()
    model(batch[0])
    _, after = snapshot(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)


def test_validation_forward_changes_nothing(batch):
    model, _ = build_model("scratch", 3, init_seed=0, init_override="random")
    before_params, before_buffers = snapshot(model)
    model.eval()
    with torch.inference_mode():
        model(batch[0])
    after_params, after_buffers = snapshot(model)
    assert all(torch.equal(before_params[n], after_params[n]) for n in before_params)
    assert all(torch.equal(before_buffers[n], after_buffers[n]) for n in before_buffers)


def test_prototypes_train_nothing():
    model, _ = build_model("prototypes", 3, init_seed=0, init_override="random")
    assert not any(p.requires_grad for p in model.parameters())
    assert optimizer_groups(model, "prototypes", OptimizerSettings()) == []
    enter_train_mode(model, "prototypes")
    assert not model.training


def test_learning_rates_per_group():
    settings = OptimizerSettings()
    model, _ = build_model("partial", 3, init_seed=0, init_override="random")
    groups = {g["name"]: g["lr"] for g in optimizer_groups(model, "partial", settings)}
    assert groups == {"layer4": settings.lr_layer4, "head": settings.lr_head}
    model, _ = build_model("scratch", 3, init_seed=0, init_override="random")
    assert [g["lr"] for g in optimizer_groups(model, "scratch", settings)] == [settings.lr_scratch]
