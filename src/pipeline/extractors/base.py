"""Base types shared by all extractor backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from PIL import Image


@dataclass
class PersonRecord:
    barcode_number: str
    name: str
    user_id: str
    category: str  # "A" | "B" | "C"
    field_confidences: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # aggregate (min of non-zero field confidences)

    def __post_init__(self):
        if self.confidence == 0.0 and self.field_confidences:
            vals = [v for v in self.field_confidences.values() if v > 0]
            self.confidence = min(vals) if vals else 0.5


@dataclass
class ExtractionResult:
    image_path: str
    model_name: str
    persons: list[PersonRecord]
    latency_ms: float
    routing: str  # "easy" | "hard"
    raw_output: dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, image: Image.Image, image_path: str = "") -> ExtractionResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
