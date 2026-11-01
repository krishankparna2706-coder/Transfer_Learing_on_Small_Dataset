"""The model specification stored with every artifact, plus safe save and load.

A model is always rebuilt from its saved spec: architecture, encoder weights identity, class
order, evaluation transform and method parameters. It is never rebuilt from UI settings. Loading
checks the spec against the requested context, and a dataset, class-map or transform mismatch is
refused. Learned checkpoints load with ``torch.load(weights_only=True)``. Prototype files are
plain ``.npz`` arrays, read with ``allow_pickle=False``.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fewshotlab import PROTOCOL_ID
from fewshotlab.errors import FewShotLabError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import atomic_write_bytes
from fewshotlab.models.backbone import EMBEDDING_DIM, ResNet18Classifier, build_resnet18

SPEC_SCHEMA_VERSION = 1
LINEAR_FORMAT = "fewshotlab-linear-checkpoint-v1"
LEARNED_METHODS = ("scratch", "frozen", "partial")


class IncompatibleModelError(FewShotLabError):
    """A stored model does not match the dataset, class map or transform it is used with."""


@dataclass(frozen=True)
class ModelSpec:
    method: str
    dataset: str
    class_names: tuple[str, ...]
    class_map_sha256: str
    eval_transform_id: str
    eval_transform: dict[str, Any]
    encoder: dict[str, Any]  # architecture, weights_id, url, filename, sha256, init_seed
    method_params: dict[str, Any]
    support_sha256: str | None = None
    run_id: str | None = None
    is_fixture: bool = False
    protocol_id: str = PROTOCOL_ID
    embedding_dim: int = EMBEDDING_DIM
    schema_version: int = SPEC_SCHEMA_VERSION

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["class_names"] = list(self.class_names)
        return json.loads(json.dumps(data))  # tuples -> lists; JSON-safe

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ModelSpec:
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise IncompatibleModelError(f"model spec has unknown fields {unknown}")
        if data.get("schema_version") != SPEC_SCHEMA_VERSION:
            raise IncompatibleModelError(
                f"model spec schema {data.get('schema_version')} is not {SPEC_SCHEMA_VERSION}"
            )
        return cls(**{**data, "class_names": tuple(data["class_names"])})

    def check_compatible(
        self,
        *,
        dataset: str | None = None,
        class_map_sha256: str | None = None,
        class_names: tuple[str, ...] | None = None,
        eval_transform_id: str | None = None,
        method: str | None = None,
    ) -> None:
        problems = []
        if dataset is not None and dataset != self.dataset:
            problems.append(f"model dataset {self.dataset!r} != requested {dataset!r}")
        if class_map_sha256 is not None and class_map_sha256 != self.class_map_sha256:
            problems.append("class map differs (different class order or names)")
        if class_names is not None and tuple(class_names) != self.class_names:
            problems.append("class names differ")
        if eval_transform_id is not None and eval_transform_id != self.eval_transform_id:
            problems.append(
                f"evaluation transform {self.eval_transform_id} != requested {eval_transform_id}"
            )
        if method is not None and method != self.method:
            problems.append(f"method {self.method!r} != requested {method!r}")
        if problems:
            raise IncompatibleModelError("model is incompatible: " + "; ".join(problems))


def save_linear(path: str | Path, model: ResNet18Classifier, spec: ModelSpec) -> str:
    """Save a learned model (state dict + spec). Returns the file's SHA-256."""
    if spec.method not in LEARNED_METHODS:
        raise ValueError(f"{spec.method!r} is not a learned method")
    if model.num_classes != spec.num_classes:
        raise ValueError("model head size does not match the spec's class count")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": LINEAR_FORMAT,
        "spec_json": json.dumps(spec.to_json(), sort_keys=True),
        "state_dict": {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()},
    }
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def load_linear(path: str | Path, **expected: Any) -> tuple[ResNet18Classifier, ModelSpec]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != LINEAR_FORMAT:
        raise IncompatibleModelError(f"{path} is not a FewShotLab linear checkpoint")
    spec = ModelSpec.from_json(json.loads(payload["spec_json"]))
    spec.check_compatible(**expected)
    model = ResNet18Classifier(spec.num_classes)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, spec


def save_prototypes(path: str | Path, prototypes: torch.Tensor, spec: ModelSpec) -> str:
    if spec.method != "prototypes":
        raise ValueError("save_prototypes needs a prototypes spec")
    if tuple(prototypes.shape) != (spec.num_classes, spec.embedding_dim):
        raise ValueError(f"prototype matrix shape {tuple(prototypes.shape)} does not match spec")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        prototypes=prototypes.detach().to("cpu").numpy().astype(np.float32),
        class_names=np.array(spec.class_names),
        spec_json=np.array(json.dumps(spec.to_json(), sort_keys=True)),
    )
    atomic_write_bytes(path, buffer.getvalue())
    return sha256_file(path)


def build_encoder_from_spec(
    spec: ModelSpec, *, weights_dir: Path | None = None, allow_download: bool = False
) -> ResNet18Classifier:
    """Rebuild the encoder recorded in ``spec`` and verify the weights identity."""
    encoder = spec.encoder
    init_seed = int(encoder.get("init_seed", 0))
    if encoder.get("weights_id") == "none":
        model, _ = build_resnet18(spec.num_classes, init="random", init_seed=init_seed)
    else:
        model, info = build_resnet18(
            spec.num_classes,
            init="imagenet",
            init_seed=init_seed,
            weights_dir=weights_dir,
            allow_download=allow_download,
        )
        if info.sha256 != encoder.get("sha256"):
            raise IncompatibleModelError(
                "the cached ImageNet weights differ from the weights recorded in the model spec"
            )
    model.eval()
    return model


def load_prototypes(
    path: str | Path,
    *,
    weights_dir: Path | None = None,
    allow_download: bool = False,
    **expected: Any,
) -> tuple[ResNet18Classifier, torch.Tensor, ModelSpec]:
    with np.load(Path(path), allow_pickle=False) as data:
        spec = ModelSpec.from_json(json.loads(str(data["spec_json"])))
        prototypes = torch.from_numpy(np.array(data["prototypes"], dtype=np.float32))
        names = tuple(str(x) for x in data["class_names"])
    if names != spec.class_names:
        raise IncompatibleModelError("prototype class order does not match its spec")
    spec.check_compatible(**expected)
    model = build_encoder_from_spec(spec, weights_dir=weights_dir, allow_download=allow_download)
    return model, prototypes, spec
