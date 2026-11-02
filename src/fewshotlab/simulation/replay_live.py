"""Live inference on the sealed NEU replay (P7), and the timing protocol it is measured with.

- **Model:** the sealed demo checkpoint. Before it loads, it is identified against the checkpoint
  hash in the sealed plan (``engine.predict.model_ref``).
- **Input:** each verified replay derivative (``SealedReplay.derivative`` checks its sealed
  checksum). It goes through the evaluation contract exactly as in the sealed robustness
  evaluation (``corruptions.to_model_input``) and is scored by the stored model's own code.
- **Check:** every live result is compared with the stored sealed prediction for the same image.
- **Timing** (document 02, p.16):
  - **forward latency:** batch size 1, a prepared tensor, evaluation mode, no gradients; 20
    warm-up requests, then 200 timed ones. The device is synchronized before and after each
    timed region, and every sample is kept.
  - **end-to-end latency:** decoding, corruption, preprocessing, device transfer, the forward
    pass and result formatting. Browser rendering is excluded.
  - **cold start:** identifying and loading the checkpoint, measured separately.
- **Playback:** its cadence is a presentation setting, never a speed figure.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from fewshotlab.data.transforms import build_transform
from fewshotlab.engine import predict
from fewshotlab.engine.inference import eval_loader
from fewshotlab.paths import ProjectPaths
from fewshotlab.simulation import corruptions, review
from fewshotlab.simulation.sealed_replay import ReplayItem, SealedReplay

WARMUP_REQUESTS = 20
TIMED_REQUESTS = 200


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def environment(device: str) -> dict[str, Any]:
    """What a latency figure depends on: device, synchronization, threads, precision."""
    sync = {"cuda": "torch.cuda.synchronize()", "mps": "torch.mps.synchronize()"}
    return {
        "device": device,
        "synchronization": sync.get(device.split(":")[0], "none needed (CPU runs synchronously)"),
        "cpu_threads": torch.get_num_threads(),
        "precision": "float32",
        "torch": torch.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "power_mode": "not recorded",
        "timer": "time.perf_counter (monotonic, high resolution)",
    }


def describe(samples_ms: list[float]) -> dict[str, Any]:
    """Median and 95th percentile (linear interpolation), with every sample kept."""
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(values.size),
        "median_ms": float(np.median(values)) if values.size else None,
        "p95_ms": float(np.percentile(values, 95)) if values.size else None,
        "min_ms": float(values.min()) if values.size else None,
        "max_ms": float(values.max()) if values.size else None,
        "samples_ms": [round(float(v), 4) for v in values],
    }


class _One(Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.tensor, -1, index


@dataclass(frozen=True)
class LiveResult:
    item: ReplayItem
    pred_class_id: int
    max_score: float
    review: bool
    preprocess_ms: float  # decode, corrupt (checksum-verified), evaluation transform
    forward_ms: float  # one synchronized scoring call: transfer, forward pass, scores to host
    end_to_end_ms: float  # everything above plus the review decision and result formatting

    @property
    def score_difference(self) -> float:
        return abs(self.max_score - self.item.max_score)

    @property
    def matches_stored(self) -> bool:
        return (
            self.pred_class_id == self.item.pred_class_id
            and self.score_difference <= predict.SCORE_TOLERANCE
        )


class LiveReplay:
    """The sealed demo checkpoint scoring verified replay images, one at a time."""

    def __init__(self, view: SealedReplay, model: predict.LoadedModel, cold_start_ms: float):
        self.view = view
        self.model = model
        self.cold_start_ms = cold_start_ms
        self.transform = build_transform(view.identity["dataset"], "eval")

    @property
    def device(self) -> str:
        return self.model.device

    @classmethod
    def load(cls, view: SealedReplay, paths: ProjectPaths, device: str) -> LiveReplay:
        """Identify the sealed checkpoint by its sealed hash, then load it (timed: cold start)."""
        identity = view.identity
        start = time.perf_counter()
        ref = predict.model_ref(
            paths,
            identity["run_id"],
            identity["attempt_id"],
            sealed_artifact_sha256=identity["checkpoint_sha256"],
        )
        model = predict.load_model(paths, ref, device)
        synchronize(device)
        return cls(view, model, (time.perf_counter() - start) * 1000.0)

    def prepare(self, item: ReplayItem) -> torch.Tensor:
        return corruptions.to_model_input(self.view.derivative(item), self.transform)

    def _score(self, tensor: torch.Tensor) -> tuple[int, float]:
        outputs = self.model.stored.score(eval_loader(_One(tensor), num_workers=0))
        return int(outputs.pred[0]), float(outputs.scores[0].max())

    def infer(self, item: ReplayItem) -> LiveResult:
        synchronize(self.device)
        start = time.perf_counter()
        tensor = self.prepare(item)
        prepared = time.perf_counter()
        pred, score = self._score(tensor)
        synchronize(self.device)
        scored = time.perf_counter()
        flagged = bool(review.review_flags([score], self.view.threshold)[0])
        fields = {
            "item": item,
            "pred_class_id": pred,
            "max_score": score,
            "review": flagged,
            "preprocess_ms": (prepared - start) * 1000.0,
            "forward_ms": (scored - prepared) * 1000.0,
        }
        return LiveResult(**fields, end_to_end_ms=(time.perf_counter() - start) * 1000.0)

    def time_forward(
        self, item: ReplayItem, *, warmup: int = WARMUP_REQUESTS, timed: int = TIMED_REQUESTS
    ) -> dict[str, Any]:
        """Forward latency on one prepared tensor (batch size 1), per the timing protocol."""
        tensor = self.prepare(item)
        for _ in range(warmup):
            self._score(tensor)
        samples = []
        for _ in range(timed):
            synchronize(self.device)
            start = time.perf_counter()
            self._score(tensor)
            synchronize(self.device)
            samples.append((time.perf_counter() - start) * 1000.0)
        return {
            **describe(samples),
            "batch_size": 1,
            "warmup_requests": warmup,
            "timed_requests": timed,
            "input_shape": [1, *tensor.shape],
            "image": item.derivative_id,
            "measures": "one scoring call on a prepared tensor: host-to-device transfer, forward "
            "pass, softmax and scores back to the host",
        }
