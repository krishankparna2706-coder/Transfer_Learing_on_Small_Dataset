"""ResNet18 construction with explicit, auditable starting weights.

Two starting points exist:

* ``random``   - ``torchvision.models.resnet18(weights=None)``, initialized under a dedicated seed
                 so initialization never shares a random stream with sampling or augmentation;
* ``imagenet`` - ``ResNet18_Weights.IMAGENET1K_V1`` loaded from the project cache.

Pretrained weights are never downloaded implicitly. Without ``allow_download=True`` a missing
file raises :class:`WeightsUnavailableError` explaining how to fetch it. A fresh model is built
for every run; nothing is warm-started.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights

from fewshotlab.errors import WeightsUnavailableError
from fewshotlab.hashing import sha256_file

EMBEDDING_DIM = 512
WEIGHTS_ID = "IMAGENET1K_V1"
_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
WEIGHTS_URL: str = _WEIGHTS.url
WEIGHTS_FILENAME: str = Path(WEIGHTS_URL).name  # torchvision names files <arch>-<sha256 prefix>.pth


@dataclass(frozen=True)
class WeightsInfo:
    """Identity of the starting weights recorded in every model spec and provenance file."""

    weights_id: str  # "IMAGENET1K_V1" or "none"
    url: str | None
    filename: str | None
    sha256: str | None


NO_PRETRAINED = WeightsInfo(weights_id="none", url=None, filename=None, sha256=None)


class ResNet18Classifier(nn.Module):
    """ResNet18 encoder producing 512-d pooled features, plus a replaceable linear head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        encoder = torchvision.models.resnet18(weights=None)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.head = nn.Linear(EMBEDDING_DIM, num_classes)
        self.num_classes = num_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _check_input(x: torch.Tensor) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            f"expected an input tensor of shape [batch, 3, H, W], got {tuple(x.shape)}"
        )


def _hash_prefix(filename: str) -> str:
    return filename.rsplit(".", 1)[0].rsplit("-", 1)[-1]


def weights_path(weights_dir: Path) -> Path:
    return Path(weights_dir) / WEIGHTS_FILENAME


def ensure_imagenet_weights(weights_dir: Path, *, allow_download: bool) -> tuple[Path, WeightsInfo]:
    """Return the cached IMAGENET1K_V1 file and its identity; download only if permitted."""
    path = weights_path(weights_dir)
    if not path.is_file():
        if not allow_download:
            raise WeightsUnavailableError(
                f"ResNet18 {WEIGHTS_ID} weights are not cached at {path}. FewShotLab never "
                f"downloads implicitly: re-run the command with --allow-download to fetch "
                f"{WEIGHTS_URL} once (it is verified against its SHA-256 prefix)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            WEIGHTS_URL, str(path), hash_prefix=_hash_prefix(WEIGHTS_FILENAME), progress=True
        )
    digest = sha256_file(path)
    if not digest.startswith(_hash_prefix(WEIGHTS_FILENAME)):
        raise WeightsUnavailableError(
            f"{path} has SHA-256 {digest}, which does not match the expected prefix "
            f"{_hash_prefix(WEIGHTS_FILENAME)}; delete it and fetch it again."
        )
    return path, WeightsInfo(WEIGHTS_ID, WEIGHTS_URL, WEIGHTS_FILENAME, digest)


@contextlib.contextmanager
def seeded_init(seed: int) -> Iterator[None]:
    """Run parameter initialization on its own CPU random stream, leaving others untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def build_resnet18(
    num_classes: int,
    *,
    init: str,
    init_seed: int,
    weights_dir: Path | None = None,
    allow_download: bool = False,
) -> tuple[ResNet18Classifier, WeightsInfo]:
    """Build a fresh model. ``init`` is ``"random"`` (scratch) or ``"imagenet"`` (transfer)."""
    if init not in ("random", "imagenet"):
        raise ValueError(f"init must be 'random' or 'imagenet', got {init!r}")
    with seeded_init(init_seed):
        model = ResNet18Classifier(num_classes)
    if init == "random":
        return model, NO_PRETRAINED
    if weights_dir is None:
        raise ValueError("weights_dir is required for init='imagenet'")
    path, info = ensure_imagenet_weights(Path(weights_dir), allow_download=allow_download)
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    dropped = sorted(set(state) - set(encoder_state))
    if dropped != ["fc.bias", "fc.weight"]:
        raise WeightsUnavailableError(
            f"unexpected ResNet18 checkpoint layout; dropped keys {dropped}"
        )
    model.encoder.load_state_dict(encoder_state, strict=True)
    return model, info
