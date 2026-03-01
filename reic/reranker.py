"""
Reranker: Probability-based intent scoring.
P(tj|q, E) = M(P, q, E)tj  — xác suất intent tj cho bởi LLM M
t̂ = argmax P(tj|q, E)     — chọn intent có xác suất cao nhất
Constrained: chỉ đánh giá các intent trong E, tránh hallucination.

--- Vị trí trong bài báo (structure.md / Figure 2) ---
• Bước 3 – LLM reranking (sau Retrieval): "LLM chỉ cần phân biệt" giữa các
  intent trong E (e.g. Change Shipping Address vs Delivery Status).
• Reasoning: "Đọc query và so sánh với intent candidates để đưa ra quyết định
  intent đúng nhất"; "LLM nhận context ngắn (top-k candidates), dùng năng lực
  ngôn ngữ & suy luận để chọn" → rerank(query, candidates).
• Thứ 4 – Hallucination: "Constrained decoding", "Probability-based intent
  scoring", "LLM chỉ đánh giá xác suất cho các intent có sẵn trong candidate
  list" → chỉ output trong E.
• Kiến trúc 2: "Reasoning (LLM Reranker): LLM nhận prompt query + candidates,
  score và chọn intent tốt nhất (argmax P(t|q; E)). Constrained để tránh
  hallucination."
• Figure 2: Retrieved intent candidates → Probability calculation → Intent LLM
  → Predicted Intent (bước này là reranker).
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


class LocalLLMReranker(Reranker):
    """
    Reranker dùng LLM local ~1.5B, fine-tune được (LoRA).
    Constrained decoding: với mỗi intent t_j, forward P(q,E) + t_j → log-prob → P(tj|q,E).
    Không generate tự do, chỉ tính xác suất cho từng t_j ∈ E.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        adapter_path: str | None = None,
        device: str | None = None,
    ):
        """
        model_name: HuggingFace model ~1.5B (Qwen2, Phi-2, ...).
        adapter_path: đường dẫn LoRA/PEFT adapter đã fine-tune (optional).
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("Cần: pip install transformers torch") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model, self.adapter_path
                )
            except ImportError:
                pass  # Chạy không adapter nếu chưa cài peft
        self._model.eval()
        self._torch = torch
        if self._device is None:
            self._device = next(self._model.parameters()).device

    def _build_base_prompt(self, query: str, candidates: list[IntentCandidate]) -> str:
        lines = [
            "Given the query and the list of intents, choose the best intent.",
            f"Query: {query}",
            "Intents:",
        ]
        for c in candidates:
            lines.append(f"- {c.intent_id}: {c.name}")
        lines.append("Best intent:")
        return "\n".join(lines)

    def rerank(
        self, query: str, candidates: list[IntentCandidate]
    ) -> tuple[str, str, float, dict[str, float]]:
        if not candidates:
            return "", "", 0.0, {}
        self._ensure_loaded()
        import torch.nn.functional as F

        base_prompt = self._build_base_prompt(query, candidates)
        base_ids = self._tokenizer(
            base_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(self._device)

        # Constrained decoding: log-prob của từng intent khi append vào prompt
        log_scores = []
        with self._torch.no_grad():
            for c in candidates:
                # Target = intent_id (hoặc name) để model "dự đoán"
                target = f" {c.intent_id}"
                full = base_prompt + target
                full_ids = self._tokenizer(
                    full,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).input_ids.to(self._device)
                # Chỉ tính log-prob trên phần target (tokens sau base_prompt)
                prompt_len = base_ids.shape[1]
                target_ids = full_ids[:, prompt_len:]
                if target_ids.shape[1] == 0:
                    log_scores.append(-1e9)
                    continue
                outputs = self._model(full_ids[:, :-1])
                logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :]
                log_probs = F.log_softmax(logits, dim=-1)
                # Gather log-prob của đúng token tại mỗi vị trí
                token_log_probs = self._torch.gather(
                    log_probs, 2, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                avg_log_prob = token_log_probs.mean().item()
                log_scores.append(avg_log_prob)

        # Chuẩn hóa thành phân phối xác suất
        log_scores = [x if x > -1e8 else -1e8 for x in log_scores]
        m = max(log_scores)
        exp_scores = [math.exp(x - m) for x in log_scores]
        total = sum(exp_scores)
        P_tj = {c.intent_id: exp_scores[i] / total for i, c in enumerate(candidates)}
        best_idx = max(range(len(P_tj)), key=lambda i: list(P_tj.values())[i])
        best = candidates[best_idx]
        return best.intent_id, best.name, P_tj[best.intent_id], P_tj
