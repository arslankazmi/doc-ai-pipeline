"""Orchestrator: lazy extractor loading, sequential inference, case storage."""
from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.case_store import insert_case
from pipeline.confidence import route

# ---------------------------------------------------------------------------
# Repo root & upload dir
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent  # src/pipeline/orchestrator.py → 3 levels up → repo root
_UPLOADS_DIR = _REPO_ROOT / "data" / "uploads"

# ---------------------------------------------------------------------------
# Extractor cache (lazy)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def get_extractor(name: str):
    """Return a cached extractor instance, instantiating on first call."""
    if name in _cache:
        return _cache[name]

    if name == "qwen2vl":
        from pipeline.extractors.qwen2vl import Qwen2VLExtractor
        extractor = Qwen2VLExtractor()
    elif name == "donut":
        from pipeline.extractors.donut import DonutExtractor
        extractor = DonutExtractor()
    elif name == "paddle_trocr":
        from pipeline.extractors.paddle_trocr import PaddleTrOCRExtractor
        extractor = PaddleTrOCRExtractor()
    else:
        raise ValueError(f"Unknown extractor: {name!r}")

    _cache[name] = extractor
    return extractor


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image(image: Image.Image) -> str:
    img = image.copy()
    if img.width > 800:
        ratio = 800 / img.width
        img = img.resize((800, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def run_extraction(
    image: Image.Image,
    image_path: str,
    model_names: list[str],
) -> dict[str, Any]:
    """Run each requested model sequentially; return model_name → ExtractionResult."""
    results: dict[str, Any] = {}
    for name in model_names:
        extractor = get_extractor(name)
        results[name] = extractor.extract(image, image_path)
    return results


def process_image(
    image: Image.Image,
    image_path: str,
    model_names: list[str],
    store_hard_cases: bool = True,
) -> dict[str, Any]:
    """Run extraction, optionally store hard cases, return summary dict."""
    extraction_results = run_extraction(image, image_path, model_names)

    results_dict: dict[str, Any] = {}
    case_ids: dict[str, str | None] = {}
    any_hard = False

    image_b64: str | None = None  # encode once lazily

    for model_name, result in extraction_results.items():
        result_as_dict = dataclasses.asdict(result)
        results_dict[model_name] = result_as_dict

        confidence = result.persons[0].confidence if result.persons else 0.0
        routing = route(confidence)

        if routing == "hard" and store_hard_cases:
            any_hard = True
            if image_b64 is None:
                image_b64 = _encode_image(image)
            case_id = insert_case(
                image_path,
                image_b64,
                model_name,
                result_as_dict,
                confidence,
            )
            case_ids[model_name] = case_id
        else:
            case_ids[model_name] = None

    return {
        "results": results_dict,
        "case_ids": case_ids,
        "any_hard": any_hard,
    }
