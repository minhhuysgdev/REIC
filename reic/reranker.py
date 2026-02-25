"""
Reranker: Probability-based intent scoring.
P(tj|q, E) = M(P, q, E)tj  — xác suất intent tj cho bởi LLM M
t̂ = argmax P(tj|q, E)     — chọn intent có xác suất cao nhất
Constrained: chỉ đánh giá các intent trong E, tránh hallucination.
"""

import json
import math
import os
import re
from abc import ABC, abstractmethod

from reic.models import IntentCandidate


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Chuẩn hóa scores thành phân phối xác suất (softmax)."""
    if not scores:
        return []
    x = [s / temperature for s in scores]
    m = max(x)
    exp_x = [math.exp(v - m) for v in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]


class Reranker(ABC):
    """Base reranker - tính P(tj|q,E) và chọn argmax."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[IntentCandidate]
    ) -> tuple[str, str, float, dict[str, float]]:
        """Trả về (intent_id, intent_name, confidence, P(tj|q,E))."""
        pass


class SimilarityReranker(Reranker):
    """
    Heuristic: dùng softmax(similarity scores) làm proxy cho P(tj|q,E).
    Không cần LLM - phù hợp demo offline.
    """

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature  # nhỏ → phân biệt rõ hơn

    def rerank(
        self, query: str, candidates: list[IntentCandidate]
    ) -> tuple[str, str, float, dict[str, float]]:
        if not candidates:
            return "", "", 0.0, {}
        scores = [c.score for c in candidates]
        probs = _softmax(scores, self.temperature)
        P_tj = {c.intent_id: p for c, p in zip(candidates, probs)}
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best = candidates[best_idx]
        return best.intent_id, best.name, probs[best_idx], P_tj


class LLMReranker(Reranker):
    """
    LLM reranker: P(tj|q,E) = M(P,q,E)tj.
    Dùng OpenAI với prompt yêu cầu output probability cho từng intent.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def rerank(
        self, query: str, candidates: list[IntentCandidate]
    ) -> tuple[str, str, float, dict[str, float]]:
        if not candidates:
            return "", "", 0.0, {}
        try:
            import openai
        except ImportError:
            return SimilarityReranker().rerank(query, candidates)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return SimilarityReranker().rerank(query, candidates)

        client = openai.OpenAI(api_key=api_key)
        prompt = self._build_probability_prompt(query, candidates)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        P_tj = self._parse_probabilities(text, candidates)
        if P_tj:
            t_hat = max(P_tj, key=P_tj.get)
            best = next(c for c in candidates if c.intent_id == t_hat)
            return t_hat, best.name, P_tj[t_hat], P_tj
        # Fallback: parse intent từ text
        text_lower = text.lower()
        for c in candidates:
            if c.intent_id.lower() in text_lower or c.name.lower() in text_lower:
                P_tj = {x.intent_id: (1.0 if x.intent_id == c.intent_id else 0.0) for x in candidates}
                return c.intent_id, c.name, 1.0, P_tj
        best = candidates[0]
        P_tj = {c.intent_id: (1.0 if c.intent_id == best.intent_id else 0.0) for c in candidates}
        return best.intent_id, best.name, 0.7, P_tj

    def _build_probability_prompt(self, query: str, candidates: list[IntentCandidate]) -> str:
        ids = [c.intent_id for c in candidates]
        lines = [
            "Cho câu hỏi và danh sách intent, đánh giá xác suất P(tj|q,E) cho TỪNG intent.",
            "CHỈ chọn trong danh sách, tổng xác suất = 1.",
            "",
            f"Câu hỏi: {query}",
            "",
            "Danh sách intent:",
        ]
        for c in candidates:
            lines.append(f"- {c.intent_id}: {c.name}. Example: {c.example}")
        lines.append("")
        lines.append(
            "Trả lời JSON (không markdown): {\"intent_id\": probability, ...}\n"
            f"Ví dụ: {{\"{ids[0]}\": 0.8, \"{ids[1] if len(ids) > 1 else ids[0]}\": 0.2}}"
        )
        return "\n".join(lines)

    def _parse_probabilities(self, text: str, candidates: list[IntentCandidate]) -> dict[str, float]:
        """Parse P(tj|q,E) từ JSON output."""
        text = text.strip()
        # Bỏ markdown code block nếu có
        if "```" in text:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                text = m.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        ids = {c.intent_id for c in candidates}
        P_tj = {}
        total = 0.0
        for k, v in data.items():
            if k in ids and isinstance(v, (int, float)) and v >= 0:
                p = float(v)
                P_tj[k] = p
                total += p
        if not P_tj:
            return {}
        if total > 0:
            P_tj = {k: v / total for k, v in P_tj.items()}
        return P_tj
