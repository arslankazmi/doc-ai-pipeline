"""Layout detection: locates per-person regions within a document image.

Uses PaddleOCR/PPStructure when available; falls back to a simple
vertical-split heuristic otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

# ---------------------------------------------------------------------------
# Optional PaddleOCR import
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR as _PaddleOCR

    _PADDLE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PADDLE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class PersonRegion:
    """Bounding box for a single person's record within the image."""

    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_paddle_ocr_instance: Any = None  # lazy singleton


def _get_paddle_ocr() -> Any:
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = _PaddleOCR(
            use_angle_cls=True,
            lang="korean",  # supports Korean + Latin
            show_log=False,
        )
    return _paddle_ocr_instance


def _group_blocks_into_regions(
    blocks: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    gap_threshold: float = 0.10,
) -> list[PersonRegion]:
    """Cluster vertically adjacent text blocks into person regions.

    *blocks* is a list of ``(left, top, right, bottom)`` quads (image coords).
    Two blocks are merged into the same region if the vertical gap between
    them is smaller than *gap_threshold* × image height.
    """
    if not blocks:
        return [PersonRegion(bbox=(0, 0, img_w, img_h), confidence=0.5)]

    # Sort top-to-bottom
    sorted_blocks = sorted(blocks, key=lambda b: b[1])
    gap_px = gap_threshold * img_h

    groups: list[list[tuple[int, int, int, int]]] = [[sorted_blocks[0]]]
    for blk in sorted_blocks[1:]:
        prev_bottom = max(b[3] for b in groups[-1])
        if blk[1] - prev_bottom <= gap_px:
            groups[-1].append(blk)
        else:
            groups.append([blk])

    regions: list[PersonRegion] = []
    for grp in groups:
        l = min(b[0] for b in grp)
        t = min(b[1] for b in grp)
        r = max(b[2] for b in grp)
        b = max(b[3] for b in grp)
        regions.append(PersonRegion(bbox=(l, t, r, b), confidence=0.9))

    return regions


def _paddle_detect(image: Image.Image) -> list[PersonRegion]:
    """Run PaddleOCR text detection and group results into person regions."""
    import numpy as np

    ocr = _get_paddle_ocr()
    arr = np.array(image.convert("RGB"))
    result = ocr.ocr(arr, cls=True)

    blocks: list[tuple[int, int, int, int]] = []
    # result is [[line, ...]] where line = [box_points, (text, score)]
    for page in (result or []):
        for line in (page or []):
            if not line:
                continue
            box_points = line[0]  # 4×2 list of [x, y]
            xs = [pt[0] for pt in box_points]
            ys = [pt[1] for pt in box_points]
            blocks.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

    return _group_blocks_into_regions(blocks, image.width, image.height)


def _heuristic_detect(image: Image.Image) -> list[PersonRegion]:
    """Simple fallback: vertical split for tall images, single region otherwise."""
    w, h = image.size
    if h > 2 * w:
        mid = h // 2
        return [
            PersonRegion(bbox=(0, 0, w, mid), confidence=0.5),
            PersonRegion(bbox=(0, mid, w, h), confidence=0.5),
        ]
    return [PersonRegion(bbox=(0, 0, w, h), confidence=0.5)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_person_regions(image: Image.Image) -> list[PersonRegion]:
    """Detect 1-2 person regions within *image*.

    Uses PaddleOCR when available, otherwise falls back to the vertical-split
    heuristic.  Always returns at least one region.
    """
    if _PADDLE_AVAILABLE:
        try:
            regions = _paddle_detect(image)
            if regions:
                return regions
        except Exception:  # pragma: no cover — degrade gracefully
            pass
    return _heuristic_detect(image)


def crop_region(image: Image.Image, region: PersonRegion) -> Image.Image:
    """Return a crop of *image* corresponding to *region*.

    *region.bbox* is ``(left, top, right, bottom)``.
    """
    return image.crop(region.bbox)
