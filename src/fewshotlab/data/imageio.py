"""Safe image decoding shared by dataset adapters, audits, transforms and the app.

Every decode checks the pixel count before decoding, verifies the file, applies the EXIF
orientation tag, and raises :class:`ImageDecodeError` instead of crashing.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000


class ImageDecodeError(ValueError):
    """The input is not a decodable image within the configured limits."""


def _source(data: str | Path | bytes) -> io.BytesIO | Path:
    return io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else Path(data)


def open_image(
    data: str | Path | bytes, *, max_pixels: int = MAX_PIXELS, label: str | None = None
) -> Image.Image:
    """Open, verify and fully decode an image with EXIF orientation applied."""
    where = label or ("<bytes>" if isinstance(data, (bytes, bytearray)) else str(data))
    try:
        with Image.open(_source(data)) as probe:
            width, height = probe.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise ImageDecodeError(
                    f"{where}: {width}x{height} pixels is outside the allowed range "
                    f"(at most {max_pixels} pixels)"
                )
            image_format = probe.format
            probe.verify()
        with Image.open(_source(data)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
    except ImageDecodeError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageDecodeError(
            f"{where}: cannot decode image ({type(exc).__name__}: {exc})"
        ) from exc
    oriented.info["fewshotlab_format"] = image_format
    return oriented


def to_rgb(image: Image.Image) -> Image.Image:
    """Three-channel RGB: grayscale is replicated, an alpha channel is dropped."""
    return image if image.mode == "RGB" else image.convert("RGB")


def pixel_sha256(image: Image.Image) -> str:
    """Hash of the decoded pixels (catches re-encoded copies with identical pixels)."""
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def is_grayscale_content(image: Image.Image) -> bool:
    """True for single-channel images, or multi-channel images whose channels are identical."""
    if image.mode in ("L", "1", "I", "I;16", "F"):
        return True
    rgb = np.asarray(image.convert("RGB"))
    return bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())


def phash64(image: Image.Image) -> int:
    """64-bit DCT perceptual hash.

    The image is converted to 8-bit grayscale, resized to 32x32 (Lanczos), transformed with an
    orthonormal 2-D DCT-II, and the top-left 8x8 coefficients are compared with their median.
    """
    from scipy.fft import dctn

    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coeffs = dctn(np.asarray(gray, dtype=np.float64), type=2, norm="ortho")[:8, :8]
    bits = (coeffs > np.median(coeffs)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value
