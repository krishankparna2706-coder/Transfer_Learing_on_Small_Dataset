"""P0: strict configuration schema and the protocol file."""

from __future__ import annotations

import re

import pytest
import yaml

from fewshotlab import config as C
from fewshotlab.errors import ConfigError
from helpers import REPO_ROOT

PROTOCOL = REPO_ROOT / "configs" / "protocol-v1.yaml"


def base(**overrides):
    data = {
        "dataset": "flowers102",
        "method": "partial",
        "shots": 5,
        "support_seed": 11,
        "split_manifest": "data/manifests/flowers102-v1.csv",
        "support_manifest": "data/manifests/supports/flowers102-k5-s11.json",
    }
    data.update(overrides)
    return data


def test_protocol_file_matches_code_constants():
    protocol = C.load_protocol(PROTOCOL)
    assert protocol["protocol_id"] == "fewshotlab-v1"
    assert C.protocol_mismatches(protocol) == []


def test_protocol_drift_is_detected(tmp_path):
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    data["recipe"]["optimizer"]["lr_head"] = 0.01
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="recipe.optimizer"):
        C.load_protocol(path)


def test_defaults_are_expanded():
    cfg = C.parse_run_config(base())
    assert cfg.weights == "IMAGENET1K_V1"
    assert cfg.train_seed == 11
    assert cfg.train.max_epochs == 100
    assert cfg.optimizer.lr_layer4 == pytest.approx(1e-4)
    assert C.parse_run_config(base(method="scratch")).weights == "none"


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        C.parse_run_config(base(learning_rate=0.1))
    with pytest.raises(ConfigError, match="unknown key"):
        C.parse_run_config(base(train={"epochs": 5}))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"method": "scratch", "weights": "IMAGENET1K_V1"}, "must use weights 'none'"),
        ({"train_seed": 22}, "train_seed = support_seed"),
        ({"test_access": "open"}, "test_access must be 'locked'"),
        ({"shots": 2}, "shots must be one of"),
        ({"support_seed": 7, "train_seed": 7}, "support_seed must be one of"),
        ({"train": {"max_epochs": 5}}, "must use the v1 recipe exactly"),
        ({"precision": "float16"}, "precision must be 'float32'"),
        ({"dataset": "cifar10"}, "dataset must be one of"),
        ({"category": "fixture"}, "is_fixture must be true"),
        ({"method": "maml"}, "method must be one of"),
    ],
)
def test_protocol_rules_are_enforced(overrides, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        C.parse_run_config(base(**overrides))


def test_diagnostic_category_may_shorten_the_budget():
    cfg = C.parse_run_config(base(category="diagnostic", train={"max_epochs": 2}))
    assert cfg.train.max_epochs == 2


def test_config_hash_is_stable_and_sensitive():
    first = C.config_hash(C.parse_run_config(base()))
    assert first == C.config_hash(C.parse_run_config(base()))
    assert first != C.config_hash(C.parse_run_config(base(shots=10)))


def test_resolved_yaml_round_trips(tmp_path):
    cfg = C.parse_run_config(base())
    path = tmp_path / "config.resolved.yaml"
    path.write_text(C.resolved_yaml(cfg), encoding="utf-8")
    assert C.load_run_config(path) == cfg


def test_exponent_without_decimal_point_is_explained():
    with pytest.raises(ConfigError, match="decimal point"):
        C.parse_run_config(base(category="development", optimizer={"eps": "1e-8"}))
