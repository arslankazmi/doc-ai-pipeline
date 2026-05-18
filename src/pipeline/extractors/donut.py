"""Donut (naver-clova-ix/donut-base) extractor for document understanding."""
from __future__ import annotations

import re
import time
from typing import Any

from PIL import Image

from .base import BaseExtractor, ExtractionResult, PersonRecord

try:
    from pipeline.confidence import route, EASY_THRESHOLD
except ImportError:
    EASY_THRESHOLD = 0.85
    route = lambda c: "easy" if c >= EASY_THRESHOLD else "hard"

_MODEL_ID = "naver-clova-ix/donut-base"
_TASK_PROMPT = "<s_extraction>"


class DonutExtractor(BaseExtractor):
    _model = None
    _processor = None

    @property
    def name(self) -> str:
        return "donut"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return

        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        processor = DonutProcessor.from_pretrained(_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(_MODEL_ID, torch_dtype=dtype)
        model = model.to(device)
        model.eval()

        cls._model = model
        cls._processor = processor
        cls._device = device

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(self, image: Image.Image, image_path: str = "") -> ExtractionResult:
        t0 = time.time()

        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            return _failed_result(image_path, self.name, str(exc), (time.time() - t0) * 1000)

        try:
            self._load()
        except Exception as exc:
            return _failed_result(image_path, self.name, str(exc), (time.time() - t0) * 1000)

        model = self.__class__._model
        processor = self.__class__._processor
        device = self.__class__._device

        try:
            pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            decoder_input_ids = processor.tokenizer(
                _TASK_PROMPT,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                generated = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=model.decoder.config.max_position_embeddings,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequence = processor.batch_decode(generated.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
                processor.tokenizer.pad_token, ""
            )
            # Strip the task prompt token
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

            # Compute confidence from decoder scores
            token_confidences: list[float] = []
            for score in generated.scores:
                probs = F.softmax(score[0], dim=-1)
                token_confidences.append(float(probs.max().item()))
            mean_conf = (
                float(sum(token_confidences) / len(token_confidences))
                if token_confidences else 0.2
            )

            persons_data = _parse_donut_output(sequence)

        except Exception as exc:
            return _failed_result(image_path, self.name, str(exc), (time.time() - t0) * 1000)

        if not persons_data:
            return _failed_result(
                image_path, self.name, "no_parse",
                (time.time() - t0) * 1000,
                raw={"raw_text": sequence},
            )

        field_keys = ["barcode_number", "name", "user_id", "category"]
        person_records: list[PersonRecord] = []
        for p in persons_data:
            fc = {k: mean_conf for k in field_keys}
            rec = PersonRecord(
                barcode_number=str(p.get("barcode_number", "")),
                name=str(p.get("name", "")),
                user_id=str(p.get("user_id", "")),
                category=str(p.get("category", "")),
                field_confidences=fc,
                confidence=mean_conf,
            )
            person_records.append(rec)

        overall_conf = min(r.confidence for r in person_records) if person_records else 0.2
        routing = route(overall_conf)
        latency_ms = (time.time() - t0) * 1000

        return ExtractionResult(
            image_path=image_path,
            model_name=self.name,
            persons=person_records,
            latency_ms=latency_ms,
            routing=routing,
            raw_output={"raw_text": sequence},
        )


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _parse_donut_output(text: str) -> list[dict[str, Any]]:
    """Parse Donut XML-like output into a list of person dicts."""
    import json

    text = text.strip()

    # 1. Try direct JSON first (unlikely from Donut but handle anyway)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "persons" in data:
            return data["persons"]
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Parse XML-like tags Donut typically produces
    # Try to find multiple <s_person> blocks or a single record
    person_blocks = re.findall(r"<s_person>(.*?)</s_person>", text, re.DOTALL)
    if person_blocks:
        return [_xml_block_to_dict(b) for b in person_blocks]

    # 3. Single record — try top-level tags directly
    single = _xml_block_to_dict(text)
    if any(v for v in single.values()):
        return [single]

    # 4. Regex fallback for bare patterns in free text
    return _regex_extract_persons(text)


def _xml_block_to_dict(block: str) -> dict[str, Any]:
    """Extract field values from Donut XML-tag output."""
    fields: dict[str, Any] = {}
    for field_name in ("barcode_number", "barcode", "name", "user_id", "userid", "category"):
        m = re.search(rf"<{field_name}>(.*?)</{field_name}>", block, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            # Normalise field aliases
            canonical = {
                "barcode": "barcode_number",
                "userid": "user_id",
            }.get(field_name, field_name)
            fields[canonical] = val
    return fields


def _regex_extract_persons(text: str) -> list[dict[str, Any]]:
    """Last-resort regex extraction from free text."""
    person: dict[str, Any] = {}

    barcode_m = re.search(r"\b(\d{12})\b", text)
    if barcode_m:
        person["barcode_number"] = barcode_m.group(1)

    category_m = re.search(r"\b([ABC])\b", text)
    if category_m:
        person["category"] = category_m.group(1)

    uid_m = re.search(r"\b([A-Z]{2,4}-\d{4}-\d{4})\b", text)
    if uid_m:
        person["user_id"] = uid_m.group(1)

    # Name: longest word sequence that doesn't match numeric/category patterns
    remaining = re.sub(r"\d{12}|\b[ABC]\b|[A-Z]{2,4}-\d{4}-\d{4}", "", text).strip()
    words = [w for w in remaining.split() if re.match(r"[A-Za-z가-힣]+", w)]
    if words:
        person["name"] = " ".join(words[:4])

    return [person] if person else []


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
