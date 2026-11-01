"""P2: nearest-prototype rules checked against a hand-computed example."""

from __future__ import annotations

import math
import re

import pytest
import torch

from fewshotlab.models.prototypes import (
    EmbeddingError,
    fit_prototypes,
    normalize_embeddings,
    predict,
    scores,
    squared_distances,
)

F64 = torch.float64


def test_hand_computed_example():
    # class 0: (3,4)->(0.6,0.8), (0,2)->(0,1); class 1: (1,0)->(1,0), (2,0)->(1,0)
    support = torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]], dtype=F64)
    prototypes = fit_prototypes(support, [0, 0, 1, 1], 2)
    assert torch.allclose(prototypes, torch.tensor([[0.3, 0.9], [1.0, 0.0]], dtype=F64))
    # the class mean is not renormalized: |(0.3, 0.9)| = 0.9487
    assert math.isclose(
        float(torch.linalg.vector_norm(prototypes[0])), math.sqrt(0.9), rel_tol=1e-12
    )
    query = normalize_embeddings(torch.tensor([[1.0, 1.0]], dtype=F64))
    distances = squared_distances(query, prototypes)
    s = math.sqrt(0.5)
    expected = torch.tensor([[(s - 0.3) ** 2 + (s - 0.9) ** 2, (s - 1.0) ** 2 + s**2]], dtype=F64)
    assert torch.allclose(distances, expected)
    assert torch.allclose(distances, torch.tensor([[0.2029437252, 0.5857864376]], dtype=F64))
    assert predict(distances).tolist() == [0]
    weights = torch.exp(-expected)
    assert torch.allclose(scores(distances), weights / weights.sum())


@pytest.mark.parametrize(
    "bad, message",
    [
        ([[0.0, 0.0]], "<= 1e-12"),
        ([[1e-13, 0.0]], "<= 1e-12"),
        ([[1e-12, 0.0]], "<= 1e-12"),  # exactly epsilon is rejected too
        ([[float("nan"), 1.0]], "non-finite embedding"),
        ([[float("inf"), 1.0]], "non-finite embedding"),
    ],
)
def test_invalid_embeddings_are_rejected_not_clamped(bad, message):
    with pytest.raises(EmbeddingError, match=re.escape(message)):
        normalize_embeddings(torch.tensor(bad, dtype=F64))


def test_overflowing_norm_is_rejected():
    with pytest.raises(EmbeddingError, match="non-finite L2 norm"):
        normalize_embeddings(torch.tensor([[3e38, 3e38]], dtype=torch.float32))


def test_ties_go_to_the_lowest_class_id():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=F64)
    query = normalize_embeddings(torch.tensor([[1.0, 1.0]], dtype=F64))
    assert predict(squared_distances(query, prototypes)).tolist() == [0]


def test_every_class_needs_a_support_embedding():
    support = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=F64)
    with pytest.raises(EmbeddingError, match="class 1"):
        fit_prototypes(support, [0, 0], 2)


def test_chunked_distances_match_direct_computation():
    generator = torch.Generator().manual_seed(0)
    queries = normalize_embeddings(torch.randn(700, 16, generator=generator, dtype=F64))
    prototypes = torch.randn(5, 16, generator=generator, dtype=F64)
    direct = ((queries[:, None, :] - prototypes[None]) ** 2).sum(-1)
    assert torch.allclose(squared_distances(queries, prototypes, chunk=64), direct)
