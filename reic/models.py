"""Data models cho REIC."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentDefinition:
    """Định nghĩa một intent trong ontology phân cấp."""

    id: str
    name: str
    description: str
    path: list[str]  # e.g. ["Order-related", "Shipping", "ChangeShippingAddress"]
    examples: list[str] = field(default_factory=list)
    vertical: str = "3P"  # 3P hoặc 1P


@dataclass
class IntentCandidate:
    """Ứng cử viên intent được retrieve từ index."""

    intent_id: str
    name: str
    description: str
    path: list[str]
    example: str  # ví dụ khớp nhất
    score: float  # similarity score từ retriever


@dataclass
class ReicResult:
    """Kết quả dự đoán intent từ pipeline REIC."""

    predicted_intent: str
    intent_id: str
    path: list[str]
    confidence: float  # P(t̂|q, E)
    candidates: list[IntentCandidate]
    intent_probabilities: dict[str, float] = field(default_factory=dict)  # P(tj|q,E) ∀ tj∈E
    vertical: str = "3P"
