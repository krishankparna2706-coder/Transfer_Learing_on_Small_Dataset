"""Nearest-prototype transfer baseline.

This is not an episodically trained Prototypical Network. The rules are:

1. Reject non-finite embeddings, non-finite L2 norms, and norms <= 1e-12 (never clamp).
2. L2-normalize each support and query embedding individually.
3. Each prototype is the arithmetic mean of its class's normalized support embeddings; it is not
   normalized again.
4. Predict the class with the minimum squared Euclidean distance (ties: lowest class ID).
5. The display score is softmax(-squared distance) at scale 1 (uncalibrated).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

EPSILON = 1e-12


class EmbeddingError(ValueError):
    """An embedding violates the prototype rules (non-finite, or norm <= 1e-12)."""


def normalize_embeddings(
    embeddings: torch.Tensor, ids: Sequence[str] | None = None
) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise EmbeddingError(
            f"expected a non-empty [N, D] embedding matrix, got {tuple(embeddings.shape)}"
        )
    names = list(ids) if ids is not None else [str(i) for i in range(embeddings.shape[0])]
    finite = torch.isfinite(embeddings).all(dim=1)
    if not bool(finite.all()):
        bad = [names[i] for i in torch.nonzero(~finite).flatten().tolist()][:5]
        raise EmbeddingError(f"non-finite embedding values for {bad}")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if not bool(torch.isfinite(norms).all()):
        bad = [names[i] for i in torch.nonzero(~torch.isfinite(norms)).flatten().tolist()][:5]
        raise EmbeddingError(f"non-finite L2 norm for {bad}")
    tiny = norms <= EPSILON
    if bool(tiny.any()):
        bad = [names[i] for i in torch.nonzero(tiny).flatten().tolist()][:5]
        raise EmbeddingError(
            f"L2 norm <= {EPSILON} for {bad}; such embeddings are rejected, never clamped"
        )
    return embeddings / norms.unsqueeze(1)


def fit_prototypes(
    embeddings: torch.Tensor,
    labels: Sequence[int] | torch.Tensor,
    num_classes: int,
    ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """[C, D] matrix of class means of the individually normalized support embeddings."""
    normalized = normalize_embeddings(embeddings, ids)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.shape[0] != normalized.shape[0]:
        raise EmbeddingError("labels and embeddings have different lengths")
    prototypes = torch.empty(num_classes, normalized.shape[1], dtype=normalized.dtype)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not bool(mask.any()):
            raise EmbeddingError(f"class {class_id} has no support embedding")
        prototypes[class_id] = normalized[mask].mean(dim=0)
    return prototypes


def squared_distances(
    queries_normalized: torch.Tensor, prototypes: torch.Tensor, chunk: int = 256
) -> torch.Tensor:
    """Exact squared Euclidean distances (no ||a||^2 - 2ab + ||b||^2 expansion)."""
    parts = []
    for start in range(0, queries_normalized.shape[0], chunk):
        diff = queries_normalized[start : start + chunk, None, :] - prototypes[None, :, :]
        parts.append((diff * diff).sum(dim=-1))
    return torch.cat(parts, dim=0)


def predict(distances: torch.Tensor) -> torch.Tensor:
    """Minimum distance; ``torch.argmin`` returns the first (lowest class ID) on ties."""
    return torch.argmin(distances, dim=1)


def scores(distances: torch.Tensor) -> torch.Tensor:
    return torch.softmax(-distances, dim=1)
