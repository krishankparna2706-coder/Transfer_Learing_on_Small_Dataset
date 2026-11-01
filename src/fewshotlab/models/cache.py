"""Content-addressed embedding cache.

A cached embedding matrix is reused only when the encoder weights, the transform, the dataset
manifest, the class order and the exact ordered image list all match. A cache entry can therefore
never leak across encoders, datasets, class maps, transforms or supports. A mismatch is a miss,
never a silent reuse.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fewshotlab.hashing import sha256_json
from fewshotlab.io_utils import atomic_write_bytes


def cache_metadata(
    *,
    encoder_sha256: str,
    transform_id: str,
    manifest_sha256: str,
    class_map_sha256: str,
    image_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "encoder_sha256": encoder_sha256,
        "transform_id": transform_id,
        "manifest_sha256": manifest_sha256,
        "class_map_sha256": class_map_sha256,
        "image_ids_sha256": sha256_json(list(image_ids)),
        "n_images": len(image_ids),
    }


def cache_key(metadata: dict[str, Any]) -> str:
    return sha256_json(metadata)


class EmbeddingCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def load(self, metadata: dict[str, Any]) -> np.ndarray | None:
        path = self.path(cache_key(metadata))
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(data["metadata_json"]))
            if stored != metadata:
                return None
            return np.array(data["embeddings"])

    def save(self, metadata: dict[str, Any], embeddings: np.ndarray) -> Path:
        if embeddings.shape[0] != metadata["n_images"]:
            raise ValueError("embedding rows do not match the image list")
        buffer = io.BytesIO()
        np.savez(
            buffer,
            embeddings=embeddings.astype(np.float32),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        return atomic_write_bytes(self.path(cache_key(metadata)), buffer.getvalue())
