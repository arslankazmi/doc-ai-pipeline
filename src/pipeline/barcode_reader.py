"""Barcode reading utilities using pyzbar with graceful fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

try:
    import numpy as np
    from pyzbar import pyzbar as _pyzbar

    _PYZBAR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYZBAR_AVAILABLE = False

if TYPE_CHECKING:
    pass


@dataclass
class BarcodeResult:
    data: str
    bbox: tuple[int, int, int, int]  # (left, top, width, height)


def read_barcodes(image: Image.Image) -> list[BarcodeResult]:
    """Decode all barcodes in *image* using pyzbar.

    Returns an empty list if pyzbar is not installed or no barcodes are found.
    Only Code128 symbology is targeted, but all detected barcodes are returned
    so callers can filter further if needed.
    """
    if not _PYZBAR_AVAILABLE:
        return []

    # pyzbar works best with an L or RGB numpy array.
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)

    decoded = _pyzbar.decode(arr)
    results: list[BarcodeResult] = []
    for obj in decoded:
        rect = obj.rect  # pyzbar.Decoded.rect is a Rect namedtuple: left, top, width, height
        results.append(
            BarcodeResult(
                data=obj.data.decode("utf-8", errors="replace"),
                bbox=(rect.left, rect.top, rect.width, rect.height),
            )
        )
    return results


def crop_to_barcode_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding: int = 20,
) -> Image.Image:
    """Return a PIL crop of *image* expanded by *padding* pixels around *bbox*.

    *bbox* is ``(left, top, width, height)`` as returned by :class:`BarcodeResult`.
    The crop is clamped to the image boundaries.
    """
    left, top, width, height = bbox
    img_w, img_h = image.size

    x0 = max(0, left - padding)
    y0 = max(0, top - padding)
    x1 = min(img_w, left + width + padding)
    y1 = min(img_h, top + height + padding)

    return image.crop((x0, y0, x1, y1))
