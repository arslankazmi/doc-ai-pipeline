"""PaddleOCR + TrOCR hybrid extractor for handwritten text + layout detection."""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from PIL import Image

from .base import BaseExtractor, ExtractionResult, PersonRecord

try:
    from pipeline.confidence import route, EASY_THRESHOLD
except ImportError:
    EASY_THRESHOLD = 0.85
    route = lambda c: "easy" if c >= EASY_THRESHOLD else "hard"

_TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"


class PaddleTrOCRExtractor(BaseExtractor):
    _trocr_model = None
    _trocr_processor = None
    _paddle_ocr = None
    _device: str = "cpu"

    @property
    def name(self) -> str:
        return "paddle_trocr"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    @classmethod
    def _load_trocr(cls) -> None:
        if cls._trocr_model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
        model = model.to(device)
        model.eval()

        cls._trocr_model = model
        cls._trocr_processor = processor
        cls._device = device

    @classmethod
    def _load_paddle(cls) -> bool:
        """Try to load PaddleOCR. Returns True if successful."""
        if cls._paddle_ocr is not None:
            return True
        try:
            from paddleocr import PaddleOCR  # type: ignore
            cls._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(self, image: Image.Image, image_path: str = "") -> ExtractionResult:
        t0 = time.time()

        try:
            import torch
        except ImportError as exc:
            return _failed_result(image_path, self.name, str(exc), (time.time() - t0) * 1000)

        # Load TrOCR (required)
        try:
            self._load_trocr()
        except Exception as exc:
            return _failed_result(image_path, self.name, str(exc), (time.time() - t0) * 1000)

        image_rgb = image.convert("RGB")
        np_image = np.array(image_rgb)

        # ------------------------------------------------------------------
        # Stage 1: PaddleOCR text detection (optional)
        # ------------------------------------------------------------------
        paddle_available = self._load_paddle()
        paddle_regions: list[tuple[Image.Image, float]] = []   # (crop, confidence)
        paddle_texts: list[tuple[str, float]] = []             # (text, confidence)

        if paddle_available:
            try:
                ocr_result = self.__class__._paddle_ocr.ocr(np_image, cls=True)
                # ocr_result is list of list; each inner item: [bbox, (text, conf)]
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        bbox, (text, conf) = line[0], line[1]
                        paddle_texts.append((text, float(conf)))
                        # Crop region for TrOCR refinement
                        xs = [pt[0] for pt in bbox]
                        ys = [pt[1] for pt in bbox]
                        x0, y0 = int(min(xs)), int(min(ys))
                        x1, y1 = int(max(xs)), int(max(ys))
                        # Add small padding
                        x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
                        x1, y1 = min(np_image.shape[1], x1 + 4), min(np_image.shape[0], y1 + 4)
                        crop = image_rgb.crop((x0, y0, x1, y1))
                        paddle_regions.append((crop, float(conf)))
            except Exception:
                paddle_available = False

        # ------------------------------------------------------------------
        # Stage 2: TrOCR for handwritten text
        # ------------------------------------------------------------------
        trocr_texts: list[tuple[str, float]] = []

        regions_to_process: list[Image.Image]
        if paddle_regions:
            regions_to_process = [r for r, _ in paddle_regions]
        else:
            # No PaddleOCR — try to split image into top/bottom halves
            # in case there are 2 records stacked vertically
            h = image_rgb.height
            top_half = image_rgb.crop((0, 0, image_rgb.width, h // 2))
            bottom_half = image_rgb.crop((0, h // 2, image_rgb.width, h))
            regions_to_process = [top_half, bottom_half, image_rgb]

        model = self.__class__._trocr_model
        processor = self.__class__._trocr_processor
        device = self.__class__._device

        for region in regions_to_process:
            try:
                pixel_values = processor(
                    images=region, return_tensors="pt"
                ).pixel_values.to(device)

                with torch.no_grad():
                    generated = model.generate(
                        pixel_values,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                text = processor.batch_decode(
                    generated.sequences, skip_special_tokens=True
                )[0].strip()

                # sequences_scores is log-prob sum; convert to confidence
                if hasattr(generated, "sequences_scores") and generated.sequences_scores is not None:
                    log_prob = float(generated.sequences_scores[0].item())
                    # Normalise to [0, 1] heuristically: clamp log-prob in [-10, 0]
                    conf = max(0.0, min(1.0, 1.0 + log_prob / 10.0))
                else:
                    conf = 0.5

                if text:
                    trocr_texts.append((text, conf))
            except Exception:
                continue

        # Merge PaddleOCR + TrOCR text pools
        all_texts: list[tuple[str, float]] = []
        if paddle_texts:
            all_texts.extend(paddle_texts)
        all_texts.extend(trocr_texts)

        if not all_texts:
            return _failed_result(
                image_path, self.name, "no_text_detected", (time.time() - t0) * 1000
            )

        # ------------------------------------------------------------------
        # Stage 3: Field assignment and PersonRecord building
        # ------------------------------------------------------------------
        persons = _assign_fields_to_persons(all_texts)
        if not persons:
            persons = [
                PersonRecord(
                    barcode_number="", name="", user_id="", category="",
                    field_confidences={}, confidence=0.2,
                )
            ]

        mean_conf = (
            sum(c for _, c in all_texts) / len(all_texts) if all_texts else 0.2
        )
        overall_conf = min(p.confidence for p in persons) if persons else mean_conf
        routing = route(overall_conf)
        latency_ms = (time.time() - t0) * 1000

        return ExtractionResult(
            image_path=image_path,
            model_name=self.name,
            persons=persons,
            latency_ms=latency_ms,
            routing=routing,
            raw_output={
                "paddle_texts": paddle_texts,
                "trocr_texts": trocr_texts,
            },
        )


# ------------------------------------------------------------------
# Field assignment helpers
# ------------------------------------------------------------------

_BARCODE_RE = re.compile(r"\b(\d{12})\b")
_CATEGORY_RE = re.compile(r"\b([ABC])\b")
_USER_ID_RE = re.compile(r"\b([A-Z]{2,4}-\d{4}-\d{4,6})\b")


def _assign_fields_to_persons(
    texts: list[tuple[str, float]]
) -> list[PersonRecord]:
    """
    Heuristically assign extracted text tokens to person fields.
    Supports 1-2 persons per image by grouping unmatched tokens.
    """
    # Pool all text strings
    combined_text = " ".join(t for t, _ in texts)
    mean_conf = sum(c for _, c in texts) / len(texts) if texts else 0.2

    barcode_matches = _BARCODE_RE.findall(combined_text)
    uid_matches = _USER_ID_RE.findall(combined_text)
    category_matches = _CATEGORY_RE.findall(combined_text)

    # Remove matched tokens from remainder to find names
    remainder = combined_text
    for pat in [_BARCODE_RE, _USER_ID_RE, _CATEGORY_RE]:
        remainder = pat.sub("", remainder)
    name_words = [w for w in remainder.split() if re.match(r"[A-Za-z가-힣]{2,}", w)]

    # Determine how many person records to create
    n_persons = max(
        len(barcode_matches) if barcode_matches else 1,
        len(uid_matches) if uid_matches else 1,
        1,
    )
    n_persons = min(n_persons, 2)  # cap at 2 per image spec

    # Split name words evenly across persons
    names = _split_names(name_words, n_persons)

    persons: list[PersonRecord] = []
    for i in range(n_persons):
        barcode = barcode_matches[i] if i < len(barcode_matches) else ""
        uid = uid_matches[i] if i < len(uid_matches) else ""
        category = category_matches[i] if i < len(category_matches) else ""
        name = names[i] if i < len(names) else ""

        fc = {
            "barcode_number": mean_conf if barcode else 0.0,
            "name": mean_conf if name else 0.0,
            "user_id": mean_conf if uid else 0.0,
            "category": mean_conf if category else 0.0,
        }
        rec = PersonRecord(
            barcode_number=barcode,
            name=name,
            user_id=uid,
            category=category,
            field_confidences=fc,
            confidence=mean_conf,
        )
        persons.append(rec)

    return persons


def _split_names(words: list[str], n: int) -> list[str]:
    """Split a flat list of name words across n persons."""
    if n <= 1:
        return [" ".join(words)]
    chunk = max(1, len(words) // n)
    result = []
    for i in range(n):
        start = i * chunk
        end = start + chunk if i < n - 1 else len(words)
        result.append(" ".join(words[start:end]))
    return result


def _failed_result(
    image_path: str,
    model_name: str,
    reason: str,
    latency_ms: float,
    raw: dict[str, Any] | None = None,
) -> ExtractionResult:
    fallback = PersonRecord(
        barcode_number="", name="", user_id="", category="",
        field_confidences={}, confidence=0.2,
    )
    return ExtractionResult(
        image_path=image_path,
        model_name=model_name,
        persons=[fallback],
        latency_ms=latency_ms,
        routing="hard",
        raw_output=raw or {"error": reason},
    )
