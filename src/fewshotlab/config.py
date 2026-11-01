"""Strict, validated configuration for FewShotLab runs and the study protocol.

* Unknown keys are rejected, so a typo can never silently fall back to a default.
* :func:`parse_run_config` expands every default; the resolved form is what gets hashed and
  saved as ``config.resolved.yaml``.
* Protocol v1 constants (recipe, shots, seeds, counts) are defined once here.
  ``configs/protocol-v1.yaml`` is checked against them, so the two cannot drift apart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fewshotlab import PROTOCOL_ID
from fewshotlab.errors import ConfigError
from fewshotlab.hashing import sha256_json

DATASETS: tuple[str, ...] = ("flowers102", "neu_cls")
NUM_CLASSES: dict[str, int] = {"flowers102": 102, "neu_cls": 6}
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "flowers102": {"train": 1020, "val": 1020, "test": 6149},
    "neu_cls": {"train": 1080, "val": 360, "test": 360},  # nominal; may change after audit
}
NEU_SPLIT_SEED = 2026
METHODS: tuple[str, ...] = ("scratch", "frozen", "partial", "prototypes")
SHOTS: tuple[int, ...] = (1, 3, 5, 10)
SUPPORT_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
DEVICES: tuple[str, ...] = ("auto", "cpu", "mps", "cuda")
#: core/mvp runs must use the v1 recipe exactly; development/diagnostic runs are separate
#: ledger categories and may shorten budgets; fixture runs are software tests only.
CATEGORIES: tuple[str, ...] = ("core", "mvp", "development", "diagnostic", "fixture")
RECIPE_BOUND_CATEGORIES: tuple[str, ...] = ("core", "mvp")
IMAGENET_WEIGHTS = "IMAGENET1K_V1"
NO_WEIGHTS = "none"


@dataclass(frozen=True)
class TrainSettings:
    max_epochs: int = 100
    patience: int = 15
    batch_size_cap: int = 32  # batch size = min(cap, number of support images)
    num_workers: int = 0  # execution detail, not part of the recipe


@dataclass(frozen=True)
class OptimizerSettings:
    name: str = "adamw"
    weight_decay: float = 1e-4
    betas: tuple[float, ...] = (0.9, 0.999)
    eps: float = 1e-8
    lr_scratch: float = 1e-3  # scratch: every parameter
    lr_head: float = 1e-3  # frozen and partial: the replacement classifier
    lr_layer4: float = 1e-4  # partial: layer4, including its BatchNorm affine parameters


@dataclass(frozen=True)
class SelectionSettings:
    metric: str = "macro_f1"
    tie_breakers: tuple[str, ...] = ("lower_val_loss", "earlier_epoch")
    patience_min_delta: float = 0.001  # resets the patience reference only; never ranks


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    method: str
    shots: int
    support_seed: int
    train_seed: int
    split_manifest: str
    support_manifest: str
    weights: str
    protocol_id: str = PROTOCOL_ID
    backbone: str = "resnet18"
    train: TrainSettings = field(default_factory=TrainSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    device: str = "auto"
    precision: str = "float32"
    test_access: str = "locked"
    category: str = "core"
    is_fixture: bool = False


# --------------------------------------------------------------------------- parsing helpers


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(data: Mapping[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed keys: {sorted(allowed)}")
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")


def _typed(value: Any, kind: type, where: str) -> Any:
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{where}: expected a number, got {value!r} "
                "(YAML 1.1 needs a decimal point in exponents, e.g. 1.0e-08)"
            )
        return float(value)
    if kind is str:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
        return value
    raise TypeError(f"unsupported config type {kind}")


def _parse_section(cls: type, data: Any, where: str) -> Any:
    """Parse a flat section dataclass whose fields are int/float/str or tuples of them."""
    mapping = _require_mapping(data, where)
    allowed = {f.name for f in dataclasses.fields(cls)}
    _check_keys(mapping, allowed, set(), where)
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in mapping:
            continue
        default = getattr(defaults, f.name)
        value = mapping[f.name]
        key_where = f"{where}.{f.name}"
        if isinstance(default, tuple):
            if not isinstance(value, (list, tuple)) or not value:
                raise ConfigError(f"{key_where}: expected a non-empty list")
            kwargs[f.name] = tuple(_typed(v, type(default[0]), key_where) for v in value)
        else:
            kwargs[f.name] = _typed(value, type(default), key_where)
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    """Dataclasses/tuples/mappings -> plain JSON/YAML-safe structures."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return data


# --------------------------------------------------------------------------- run configs

_RUN_REQUIRED = {"dataset", "method", "shots", "support_seed", "split_manifest", "support_manifest"}
_RUN_OPTIONAL = {
    "protocol_id",
    "backbone",
    "weights",
    "train_seed",
    "train",
    "optimizer",
    "selection",
    "device",
    "precision",
    "test_access",
    "category",
    "is_fixture",
}


def parse_run_config(data: Mapping[str, Any], where: str = "config") -> RunConfig:
    """Validate a raw mapping and expand every default into a :class:`RunConfig`."""
    mapping = _require_mapping(data, where)
    _check_keys(mapping, _RUN_REQUIRED | _RUN_OPTIONAL, _RUN_REQUIRED, where)
    method = _typed(mapping["method"], str, f"{where}.method")
    support_seed = _typed(mapping["support_seed"], int, f"{where}.support_seed")
    default_weights = NO_WEIGHTS if method == "scratch" else IMAGENET_WEIGHTS
    cfg = RunConfig(
        dataset=_typed(mapping["dataset"], str, f"{where}.dataset"),
        method=method,
        shots=_typed(mapping["shots"], int, f"{where}.shots"),
        support_seed=support_seed,
        train_seed=_typed(mapping.get("train_seed", support_seed), int, f"{where}.train_seed"),
        split_manifest=_typed(mapping["split_manifest"], str, f"{where}.split_manifest"),
        support_manifest=_typed(mapping["support_manifest"], str, f"{where}.support_manifest"),
        weights=_typed(mapping.get("weights", default_weights), str, f"{where}.weights"),
        protocol_id=_typed(mapping.get("protocol_id", PROTOCOL_ID), str, f"{where}.protocol_id"),
        backbone=_typed(mapping.get("backbone", "resnet18"), str, f"{where}.backbone"),
        train=_parse_section(TrainSettings, mapping.get("train", {}), f"{where}.train"),
        optimizer=_parse_section(
            OptimizerSettings, mapping.get("optimizer", {}), f"{where}.optimizer"
        ),
        selection=_parse_section(
            SelectionSettings, mapping.get("selection", {}), f"{where}.selection"
        ),
        device=_typed(mapping.get("device", "auto"), str, f"{where}.device"),
        precision=_typed(mapping.get("precision", "float32"), str, f"{where}.precision"),
        test_access=_typed(mapping.get("test_access", "locked"), str, f"{where}.test_access"),
        category=_typed(mapping.get("category", "core"), str, f"{where}.category"),
        is_fixture=_typed(mapping.get("is_fixture", False), bool, f"{where}.is_fixture"),
    )
    validate_run_config(cfg, where)
    return cfg


def validate_run_config(cfg: RunConfig, where: str = "config") -> None:
    """Enforce protocol v1 rules; raise :class:`ConfigError` listing every violation."""
    errors: list[str] = []
    if cfg.protocol_id != PROTOCOL_ID:
        errors.append(f"protocol_id must be {PROTOCOL_ID!r} for this code version")
    if cfg.category not in CATEGORIES:
        errors.append(f"category must be one of {list(CATEGORIES)}")
    if cfg.is_fixture != (cfg.category == "fixture"):
        errors.append("is_fixture must be true exactly when category is 'fixture'")
    if cfg.is_fixture:
        if not cfg.dataset.startswith("fixture"):
            errors.append("fixture configs must use a dataset name starting with 'fixture'")
        if cfg.shots < 1:
            errors.append("shots must be >= 1")
    else:
        if cfg.dataset not in DATASETS:
            errors.append(f"dataset must be one of {list(DATASETS)}")
        if cfg.shots not in SHOTS:
            errors.append(f"shots must be one of {list(SHOTS)}")
        if cfg.support_seed not in SUPPORT_SEEDS:
            errors.append(f"support_seed must be one of {list(SUPPORT_SEEDS)}")
    if cfg.method not in METHODS:
        errors.append(f"method must be one of {list(METHODS)}")
    if cfg.backbone != "resnet18":
        errors.append("backbone must be 'resnet18' (other encoders need their own protocol)")
    expected_weights = NO_WEIGHTS if cfg.method == "scratch" else IMAGENET_WEIGHTS
    if cfg.weights != expected_weights:
        errors.append(f"method {cfg.method!r} must use weights {expected_weights!r}")
    if cfg.train_seed != cfg.support_seed:
        errors.append("protocol v1 fixes train_seed = support_seed (streams are derived from it)")
    if cfg.device not in DEVICES:
        errors.append(f"device must be one of {list(DEVICES)}")
    if cfg.precision != "float32":
        errors.append("precision must be 'float32' in protocol v1")
    if cfg.test_access != "locked":
        errors.append(
            "test_access must be 'locked': a config can never open the test split; "
            "only the sealed lock-test workflow can"
        )
    t, o, s = cfg.train, cfg.optimizer, cfg.selection
    if not 1 <= t.max_epochs <= 100:
        errors.append("train.max_epochs must be between 1 and 100")
    if t.patience < 1:
        errors.append("train.patience must be >= 1")
    if t.batch_size_cap < 2:
        errors.append("train.batch_size_cap must be >= 2")
    if t.num_workers < 0:
        errors.append("train.num_workers must be >= 0")
    if o.name != "adamw":
        errors.append("optimizer.name must be 'adamw'")
    if min(o.lr_scratch, o.lr_head, o.lr_layer4) <= 0 or o.weight_decay < 0 or o.eps <= 0:
        errors.append("optimizer learning rates and eps must be > 0; weight_decay >= 0")
    if len(o.betas) != 2 or not all(0.0 <= b < 1.0 for b in o.betas):
        errors.append("optimizer.betas must be two numbers in [0, 1)")
    if s.metric != "macro_f1":
        errors.append("selection.metric must be 'macro_f1'")
    if s.tie_breakers != SelectionSettings().tie_breakers:
        errors.append("selection.tie_breakers must be [lower_val_loss, earlier_epoch]")
    if s.patience_min_delta < 0:
        errors.append("selection.patience_min_delta must be >= 0")
    if cfg.category in RECIPE_BOUND_CATEGORIES:
        recipe_train = dataclasses.replace(t, num_workers=TrainSettings().num_workers)
        if recipe_train != TrainSettings() or o != OptimizerSettings() or s != SelectionSettings():
            errors.append(
                f"category {cfg.category!r} must use the v1 recipe exactly; use category "
                "'development' or 'diagnostic' for any other budget or hyperparameter"
            )
    if errors:
        raise ConfigError(f"{where}: invalid run config:\n  - " + "\n  - ".join(errors))


def load_run_config(path: str | Path) -> RunConfig:
    return parse_run_config(load_yaml_mapping(path), where=str(path))


def resolved_dict(cfg: RunConfig) -> dict[str, Any]:
    return _plain(cfg)


def resolved_yaml(cfg: RunConfig) -> str:
    return yaml.safe_dump(resolved_dict(cfg), sort_keys=True, default_flow_style=False)


def config_hash(cfg: RunConfig) -> str:
    return sha256_json(resolved_dict(cfg))


# --------------------------------------------------------------------------- protocol file

_PROTOCOL_KEYS = {
    "protocol_id",
    "backbone",
    "weights",
    "datasets",
    "shots",
    "support_seeds",
    "methods",
    "recipe",
    "metrics",
    "test_access",
}


def protocol_mismatches(data: Mapping[str, Any]) -> list[str]:
    """Differences between a protocol YAML mapping and the constants in this module."""
    problems: list[str] = []

    def expect(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{label}: file has {actual!r}, code has {expected!r}")

    expect("protocol_id", data.get("protocol_id"), PROTOCOL_ID)
    expect("backbone", data.get("backbone"), "resnet18")
    expect("weights", data.get("weights"), IMAGENET_WEIGHTS)
    expect("shots", data.get("shots"), list(SHOTS))
    expect("support_seeds", data.get("support_seeds"), list(SUPPORT_SEEDS))
    expect("methods", data.get("methods"), list(METHODS))
    expect("test_access", data.get("test_access"), "locked")
    datasets = data.get("datasets") or {}
    expect("datasets", sorted(datasets), sorted(DATASETS))
    for name in DATASETS:
        spec = datasets.get(name) or {}
        expect(f"datasets.{name}.num_classes", spec.get("num_classes"), NUM_CLASSES[name])
        expect(
            f"datasets.{name}.expected_counts", spec.get("expected_counts"), EXPECTED_COUNTS[name]
        )
    expect(
        "datasets.neu_cls.split_seed",
        (datasets.get("neu_cls") or {}).get("split_seed"),
        NEU_SPLIT_SEED,
    )
    recipe = data.get("recipe") or {}
    expect("recipe.train", recipe.get("train"), _plain(TrainSettings()))
    expect("recipe.optimizer", recipe.get("optimizer"), _plain(OptimizerSettings()))
    expect("recipe.selection", recipe.get("selection"), _plain(SelectionSettings()))
    expect("recipe.precision", recipe.get("precision"), "float32")
    metrics = data.get("metrics") or {}
    expect("metrics.primary", metrics.get("primary"), "macro_f1")
    expect("metrics.zero_division", metrics.get("zero_division"), 0)
    return problems


def load_protocol(path: str | Path) -> dict[str, Any]:
    data = load_yaml_mapping(path)
    _check_keys(data, _PROTOCOL_KEYS, _PROTOCOL_KEYS, str(path))
    problems = protocol_mismatches(data)
    if problems:
        raise ConfigError(
            f"{path} disagrees with the protocol constants in fewshotlab.config:\n  - "
            + "\n  - ".join(problems)
        )
    return data
