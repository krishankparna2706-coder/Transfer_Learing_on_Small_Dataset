"""P1: seeded random streams follow the documented SHA-256 derivation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fewshotlab.seeds import derive_seed, numpy_rng, torch_generator


def test_derive_seed_matches_documented_formula():
    expected = int.from_bytes(hashlib.sha256(b'["neu_cls","split",2026,3]').digest()[:8], "big")
    assert derive_seed("neu_cls", "split", 2026, 3) == expected


def test_streams_are_reproducible_and_independent():
    a = numpy_rng("flowers102", "support", 11, 5).permutation(10)
    b = numpy_rng("flowers102", "support", 11, 5).permutation(10)
    c = numpy_rng("flowers102", "support", 22, 5).permutation(10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    import torch

    x = torch.rand(3, generator=torch_generator("init", 11))
    y = torch.rand(3, generator=torch_generator("init", 11))
    assert torch.equal(x, y)


@pytest.mark.parametrize("label", [1.5, True, None, ("tuple",)])
def test_only_str_and_int_labels(label):
    with pytest.raises(TypeError):
        derive_seed("x", label)
