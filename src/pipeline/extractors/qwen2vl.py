"""Qwen2-VL-2B-Instruct extractor with optional LoRA from finetuning experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseExtractor, ExtractionResult, PersonRecord

try:
    from pipeline.confidence import route, EASY_THRESHOLD
except ImportError:
    EASY_THRESHOLD = 0.85
    route = lambda c: "easy" if c >= EASY_THRESHOLD else "hard"

_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/pipeline/extractors/qwen2vl.py → 4 levels up → repo root
_LORA_DIR = _REPO_ROOT / "data" / "lora_model"

_SYSTEM_PROMPT = (
    "You are a document extraction assistant. Given an image of a form, identify all person "
    "records and return a JSON object with a 'persons' array. Each person has: "
    "barcode_number (12-digit string), name (as written), user_id (alphanumeric code), "
    "category (A, B, or C)."
)
_USER_PROMPT = (
    "Extract all person records from this image. "
    "Return JSON with: barcode_number, name, user_id, category."
)


class Qwen2VLExtractor(BaseExtractor):
    _model = None
    _processor = None

    @property
    def name(self) -> str:
        return "qwen2vl"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            model = model.to(device)

        # Apply LoRA weights if they exist
        if _LORA_DIR.exists():
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(_LORA_DIR))
                model = model.merge_and_unload()
            except Exception:
                pass  # Fall back to base model silently

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

        # Build conversation
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            input_len = inputs["input_ids"].shape[1]
            completion_ids = generated.sequences[0][input_len:]
            completion_text = processor.decode(completion_ids, skip_special_tokens=True)

            # Per-token confidence: softmax over vocab → max prob at each step
            token_confidences: list[float] = []
            for score in generated.scores:
                probs = F.softmax(score[0], dim=-1)
                token_confidences.append(float(probs.max().item()))

            mean_conf = float(sum(token_confidences) / len(token_confidences)) if token_confidences else 0.2

            # Parse JSON
            persons_data = _parse_persons_json(completion_text)

        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            fallback = PersonRecord(
                barcode_number="", name="", user_id="", category="",
                field_confidences={}, confidence=0.2,
            )
            return ExtractionResult(
                image_path=image_path,
                model_name=self.name,
                persons=[fallback],
                latency_ms=latency_ms,
                routing="hard",
                raw_output={"error": str(exc)},
            )

        if not persons_data:
            latency_ms = (time.time() - t0) * 1000
            fallback = PersonRecord(
                barcode_number="", name="", user_id="", category="",
                field_confidences={}, confidence=0.2,
            )
            return ExtractionResult(
                image_path=image_path,
                model_name=self.name,
                persons=[fallback],
                latency_ms=latency_ms,
                routing="hard",
                raw_output={"raw_text": completion_text},
            )

        field_keys = ["barcode_number", "name", "user_id", "category"]
        per_field_conf = mean_conf  # equal split across fields

        person_records: list[PersonRecord] = []
        for p in persons_data:
            fc = {k: per_field_conf for k in field_keys}
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
            raw_output={"raw_text": completion_text},
        )


def _parse_persons_json(text: str) -> list[dict[str, Any]]:
    """Try to extract a persons array from raw model text."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "persons" in data:
            return data["persons"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object/array embedded in the text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                snippet = text[start:end + 1]
                data = json.loads(snippet)
                if isinstance(data, dict) and "persons" in data:
                    return data["persons"]
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

    return []


def _failed_result(
    image_path: str,
    model_name: str,
    reason: str,
    latency_ms: float,
    raw: dict[str, Any] | None = None,
) -> "ExtractionResult":
    from pipeline.extractors.base import ExtractionResult, PersonRecord
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
        raw_output={"error": reason, **(raw or {})},
    )
    return []
