"""Deterministic, labeled random streams.

Every stream is seeded from SHA-256 of a JSON-encoded label list, for example
``derive_seed("neu_cls", "split", 2026, 3)``. Python's built-in ``hash()`` is never used because
it changes between processes. Adding or changing one stream can never shift another.
"""

from __future__ import annotations

import hashlib

import numpy as np

from fewshotlab.hashing import canonical_json

SEED_DERIVATION = (
    "uint64 = first 8 bytes (big-endian) of SHA-256 over the canonical JSON label list"
)
NUMPY_BIT_GENERATOR = "numpy.random.PCG64"


def derive_seed(*labels: str | int) -> int:
    """A 64-bit seed that depends only on the given labels."""
    if not labels:
        raise ValueError("at least one label is required")
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, (str, int)):
            raise TypeError(f"seed labels must be str or int, got {type(label).__name__}")
    digest = hashlib.sha256(canonical_json(list(labels)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def numpy_rng(*labels: str | int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(*labels)))


def torch_generator(*labels: str | int):
    """A CPU ``torch.Generator`` seeded from the labels (63-bit to stay in range)."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(*labels) & ((1 << 63) - 1))
    return generator
