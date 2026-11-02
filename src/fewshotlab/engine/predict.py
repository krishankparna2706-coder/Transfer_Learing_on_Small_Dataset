"""Live predictions on images a person supplies: the one code path the app and the CLI share.

- An image is decoded by the datasets' safe decoder (``data.imageio.open_image``), transformed by
  its model's dataset evaluation transform, and scored by the stored model's own ``score``
  function (``engine.inference``), exactly as a validation image is scored.
- Nothing here trains, writes a file, keeps an uploaded image or reads a held-out test image.
- A score is a model score: the softmax of logits, or of negative squared prototype distances.
  No calibration protocol exists, so it is not a probability of being right. An image outside
  the learned classes still receives a score for every learned class.
"""

from __future__ import annotations

import io
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from fewshotlab.data.datasets import ManifestImageDataset
from fewshotlab.data.imageio import ImageDecodeError, open_image
from fewshotlab.data.manifest import ClassMap, ImageRecord, load_class_map
from fewshotlab.data.registry import class_map_path
from fewshotlab.data.transforms import build_transform
from fewshotlab.engine import artifacts
from fewshotlab.engine.inference import StoredModel, eval_loader, load_stored_model
from fewshotlab.errors import BlockedError, FewShotLabError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import read_json
from fewshotlab.models.methods import Outputs, get_method
from fewshotlab.models.spec import IncompatibleModelError, ModelSpec
from fewshotlab.paths import ProjectPaths
from fewshotlab.reporting.aggregate import verify_evidence
from fewshotlab.status import RunStatus

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 25_000_000
ALLOWED_FORMATS = ("PNG", "JPEG", "BMP", "TIFF", "WEBP")
DEFAULT_TOP_K = 5
SCORE_LABEL = "model score"
#: Largest absolute model-score difference accepted when one image is scored on two devices.
SCORE_TOLERANCE = 1e-4
_DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError)


class UploadRejected(FewShotLabError):
    """The supplied file is not an acceptable image; the message says what to supply instead."""


class ModelUnavailable(BlockedError):
    """A stored model cannot be identified or loaded; the message names the next action."""


def decode_upload(
    data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES, max_pixels: int = MAX_UPLOAD_PIXELS
) -> Image.Image:
    """Accept a file only if its content decodes as one allowed still image.

    The name and extension are ignored: the decoded content decides. The size limits are checked
    before the pixels are decoded. Messages never repeat the file name or any pixel content.
    """
    allowed = ", ".join(ALLOWED_FORMATS)
    if not data:
        raise UploadRejected(f"the file is empty; choose an image file ({allowed})")
    if len(data) > max_bytes:
        raise UploadRejected(
            f"the file is {len(data) / 2**20:.1f} MB; the limit is {max_bytes / 2**20:g} MB"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                image_format = probe.format
                frames = int(getattr(probe, "n_frames", 1) or 1)
                width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise UploadRejected(f"the image has too many pixels; the limit is {max_pixels:,}") from exc
    except _DECODE_ERRORS as exc:
        raise UploadRejected(f"the file content is not a readable image ({allowed})") from exc
    if image_format not in ALLOWED_FORMATS:
        raise UploadRejected(f"{image_format or 'this'} images are not accepted; use {allowed}")
    if frames > 1:
        raise UploadRejected("animated or multi-page images are not accepted; use one still image")
    if width * height > max_pixels:
        raise UploadRejected(
            f"the image is {width} x {height} = {width * height:,} pixels; the limit is "
            f"{max_pixels:,}"
        )
    try:
        return open_image(data, max_pixels=max_pixels, label="the uploaded image")
    except ImageDecodeError as exc:
        raise UploadRejected(str(exc)) from exc


@dataclass(frozen=True)
class ModelRef:
    """A stored run that passed every identity check and may be loaded for prediction."""

    run_id: str
    attempt_id: str
    dataset: str
    method: str
    k: int
    seed: int
    category: str
    is_fixture: bool
    artifact_file: str
    artifact_sha256: str
    class_map_sha256: str
    num_classes: int
    score_kind: str

    @property
    def label(self) -> str:
        return f"{self.dataset} / {self.method} / K={self.k} / seed {self.seed}"


def _attempt_dir(paths: ProjectPaths, run_id: str, attempt_id: str | None) -> Path:
    dirs = artifacts.attempt_dirs(paths, run_id)
    if not dirs:
        raise ModelUnavailable(
            f"run {run_id} has no stored attempt (NOT RUN): train it first with "
            "`python -m fewshotlab benchmark --config CONFIG`, or choose another cell"
        )
    if attempt_id is not None:
        chosen = artifacts.run_root(paths, run_id) / attempt_id
        if chosen not in dirs:
            raise ModelUnavailable(f"run {run_id} has no attempt {attempt_id}")
        return chosen
    done = [
        d for d in dirs if (artifacts.read_status(d) or {}).get("status") == RunStatus.COMPLETED
    ]
    if not done:
        raise ModelUnavailable(
            f"run {run_id} has no COMPLETED attempt; finish or re-run it before predicting"
        )
    return done[-1]


def model_ref(
    paths: ProjectPaths,
    run_id: str,
    attempt_id: str | None = None,
    *,
    sealed_artifact_sha256: str | None = None,
) -> ModelRef:
    """Identify a stored run for prediction, checking before anything is loaded:

    - the attempt is COMPLETED and passes its evidence check;
    - the run record and model spec are readable, with a supported schema version;
    - the checkpoint file exists and has the hash recorded for it (and, when given, the sealed
      hash);
    - the dataset's current class map is the one the model was trained with.
    """
    adir = _attempt_dir(paths, run_id, attempt_id)
    where = f"{run_id}/{adir.name}"
    status = (artifacts.read_status(adir) or {}).get("status")
    if status != RunStatus.COMPLETED:
        raise ModelUnavailable(f"{where} is {status or 'UNKNOWN'}, not COMPLETED")
    problem = verify_evidence(adir)
    if problem:
        raise ModelUnavailable(f"{where} failed its evidence check: {problem}")
    try:
        doc = read_json(adir / "val_metrics.json")
        spec = ModelSpec.from_json(read_json(adir / "model_spec.json"))
    except (OSError, ValueError, KeyError, IncompatibleModelError) as exc:
        raise ModelUnavailable(f"{where}: unreadable run record ({exc})") from exc
    artifact = adir / doc["artifact"]["file"]
    if not artifact.is_file():
        raise ModelUnavailable(
            f"the checkpoint {doc['artifact']['file']} of {where} is missing; restore the run "
            "folder or retrain the run"
        )
    digest = sha256_file(artifact)
    if digest != doc["artifact"]["sha256"]:
        raise ModelUnavailable(f"the checkpoint of {where} differs from its run record; not loaded")
    if sealed_artifact_sha256 is not None and digest != sealed_artifact_sha256:
        raise ModelUnavailable(
            f"the checkpoint of {where} differs from the sealed plan; not loaded"
        )
    try:
        cmap = load_class_map(class_map_path(paths, spec.dataset))
    except (OSError, ValueError) as exc:
        raise ModelUnavailable(
            f"the {spec.dataset} class map is missing; run `python -m fewshotlab data prepare "
            f"--dataset {spec.dataset}`"
        ) from exc
    if cmap.sha256() != spec.class_map_sha256 or cmap.names != spec.class_names:
        raise ModelUnavailable(
            f"{where} was trained with a different {spec.dataset} class map; not loaded"
        )
    return ModelRef(
        run_id=run_id,
        attempt_id=adir.name,
        dataset=spec.dataset,
        method=spec.method,
        k=int(doc["k"]),
        seed=int(doc["seed"]),
        category=str(doc["category"]),
        is_fixture=bool(doc["is_fixture"]),
        artifact_file=doc["artifact"]["file"],
        artifact_sha256=digest,
        class_map_sha256=spec.class_map_sha256,
        num_classes=spec.num_classes,
        score_kind=get_method(spec.method).score_kind,
    )


@dataclass(frozen=True)
class LoadedModel:
    """An identified run, loaded on one device with its own class map."""

    ref: ModelRef
    stored: StoredModel
    class_map: ClassMap

    @property
    def device(self) -> str:
        return self.stored.device


def load_model(paths: ProjectPaths, ref: ModelRef, device: str) -> LoadedModel:
    """Load an identified run on ``device`` with the class map and transform it was trained with.

    A missing ImageNet weights cache raises the weights error itself, which names the fetch
    command; nothing is downloaded here.
    """
    adir = artifacts.run_root(paths, ref.run_id) / ref.attempt_id
    cmap = load_class_map(class_map_path(paths, ref.dataset))
    try:
        stored = load_stored_model(adir, paths, cmap, device)
    except (IncompatibleModelError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ModelUnavailable(f"cannot load {ref.run_id}/{ref.attempt_id}: {exc}") from exc
    return LoadedModel(ref=ref, stored=stored, class_map=cmap)


@dataclass(frozen=True)
class Prediction:
    """One image's live prediction. ``scores`` are model scores for every class, by class ID."""

    run_id: str
    attempt_id: str
    dataset: str
    method: str
    device: str
    scores: tuple[float, ...]
    top: tuple[tuple[int, str, float], ...]  # (class ID, class name, model score), best first
    preprocess_ms: float  # for the whole call, which may hold several images
    inference_ms: float
    images_in_call: int

    @property
    def pred_class_id(self) -> int:
        return self.top[0][0]

    @property
    def pred_class_name(self) -> str:
        return self.top[0][1]

    @property
    def max_score(self) -> float:
        return self.top[0][2]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "dataset": self.dataset,
            "method": self.method,
            "device": self.device,
            "score_label": SCORE_LABEL,
            "predicted_class_id": self.pred_class_id,
            "predicted_class_name": self.pred_class_name,
            "top": [{"class_id": c, "class_name": n, "model_score": s} for c, n, s in self.top],
            "scores": list(self.scores),
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "images_in_call": self.images_in_call,
        }


class _Prepared(Dataset):
    """Already-transformed images; the label is unknown (-1) and never used."""

    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.tensors[index], -1, index


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _ranked(pred: int, scores: np.ndarray, names: Sequence[str], top_k: int) -> tuple:
    """The predicted class first (exactly as the evaluation decides it), then the other classes
    by model score, ties to the lower class ID."""
    order = [int(i) for i in np.lexsort((np.arange(scores.size), -scores.astype(np.float64)))]
    order.remove(pred)
    return tuple((i, names[i], float(scores[i])) for i in [pred, *order][:top_k])


def _predictions(
    model: LoadedModel, outputs: Outputs, count: int, top_k: int, timing: tuple[float, float]
) -> list[Prediction]:
    by_index = {int(i): row for row, i in enumerate(outputs.indices)}
    out = []
    for index in range(count):
        row = by_index[index]
        scores = outputs.scores[row]
        out.append(
            Prediction(
                run_id=model.ref.run_id,
                attempt_id=model.ref.attempt_id,
                dataset=model.ref.dataset,
                method=model.ref.method,
                device=model.device,
                scores=tuple(float(s) for s in scores),
                top=_ranked(int(outputs.pred[row]), scores, model.class_map.names, top_k),
                preprocess_ms=timing[0],
                inference_ms=timing[1],
                images_in_call=count,
            )
        )
    return out


def predict_images(
    model: LoadedModel, images: Sequence[Image.Image], *, top_k: int = DEFAULT_TOP_K
) -> list[Prediction]:
    """Score decoded images with a stored model (no gradients, evaluation mode, fixed batches).

    Preprocessing and inference are timed separately; device work is synchronized before the
    inference clock stops.
    """
    if not images:
        return []
    transform = build_transform(model.ref.dataset, "eval")
    start = time.perf_counter()
    tensors = [transform(image) for image in images]
    prepared = time.perf_counter()
    outputs = model.stored.score(eval_loader(_Prepared(tensors), num_workers=0))
    _synchronize(model.device)
    done = time.perf_counter()
    timing = ((prepared - start) * 1000.0, (done - prepared) * 1000.0)
    return _predictions(model, outputs, len(images), top_k, timing)


def predict_records(
    model: LoadedModel,
    records: Sequence[ImageRecord],
    data_root: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Prediction]:
    """Score training or validation images through the evaluation dataset (test is refused).

    This is the path ``evaluate --split val`` takes, so the result can be checked against the
    run's stored validation predictions. Decoding is inside the loader, so the timing is one
    combined figure (reported as inference).
    """
    if not records:
        return []
    dataset = ManifestImageDataset(records, data_root, build_transform(model.ref.dataset, "eval"))
    start = time.perf_counter()
    outputs = model.stored.score(eval_loader(dataset, num_workers=0))
    _synchronize(model.device)
    timing = (0.0, (time.perf_counter() - start) * 1000.0)
    return _predictions(model, outputs, len(records), top_k, timing)


def stored_validation_prediction(
    paths: ProjectPaths, ref: ModelRef, image_id: str
) -> dict[str, Any] | None:
    """The prediction the run's own validation evaluation stored for one image, if any."""
    adir = artifacts.run_root(paths, ref.run_id) / ref.attempt_id
    stored = artifacts.read_predictions(adir / "predictions_val.csv")
    try:
        row = stored["image_ids"].index(image_id)
    except ValueError:
        return None
    return {
        "pred_class_id": int(stored["pred"][row]),
        "true_class_id": int(stored["true"][row]),
        "scores": [float(s) for s in stored["scores"][row]],
    }


def compare_scores(live: Prediction, stored: dict[str, Any]) -> dict[str, Any]:
    """How a live prediction relates to a stored one for the same image and model."""
    difference = float(
        np.max(np.abs(np.asarray(live.scores, dtype=np.float64) - np.asarray(stored["scores"])))
    )
    return {
        "same_class": live.pred_class_id == stored["pred_class_id"],
        "max_abs_score_difference": difference,
        "tolerance": SCORE_TOLERANCE,
        "within_tolerance": difference <= SCORE_TOLERANCE,
    }
