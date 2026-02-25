"""
Qwen 1.5B Baseline - LLM nhỏ, zero-shot / few-shot.
Tương tự Claude Zero-shot, Claude Few-shot trong paper (Section 5.2).
"""

import json
import re

from reic.models import IntentDefinition


class QwenClassifier:
    """
    Qwen 1.5B (hoặc Qwen2-1.5B) baseline.
    - Zero-shot: prompt với danh sách intent
    - Few-shot: thêm examples (query, intent) vào prompt
    Constrained: chỉ chọn trong danh sách, tránh hallucination.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        few_shot_examples: list[tuple[str, str]] | None = None,
    ):
        """
        few_shot_examples: [(query, intent_id), ...] tối đa ~20
        """
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples or []
        self._model = None
        self._tokenizer = None
        self._intents: list[IntentDefinition] = []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("Cần: pip install transformers torch accelerate") from e
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self._model.eval()
        self._torch = torch

    def fit(self, intents: list[IntentDefinition]) -> None:
        """Đăng ký danh sách intent."""
        self._intents = [i for i in intents if i.examples]
        self._id2intent = {i.id: i for i in self._intents}

    def predict(
        self, query: str, top_k: int = 5
    ) -> tuple[str, str, float, dict[str, float]]:
        """
        Zero-shot / Few-shot: generate intent từ prompt.
        Parse output → (intent_id, intent_name, confidence, P_tj).
        """
        if not self._intents:
            return "", "", 0.0, {}

        self._ensure_loaded()
        prompt = self._build_prompt(query)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if self._model.device.type != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        text = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        text = text.strip()
        intent_id = self._parse_intent(text)
        if intent_id and intent_id in self._id2intent:
            best = self._id2intent[intent_id]
            P_tj = {i.id: (1.0 if i.id == intent_id else 0.0) for i in self._intents}
            return intent_id, best.name, 1.0, P_tj
        # Fallback: top intent
        best = self._intents[0]
        P_tj = {i.id: (1.0 if i.id == best.id else 0.0) for i in self._intents}
        return best.id, best.name, 0.5, P_tj

    def _build_prompt(self, query: str) -> str:
        lines = [
            "Chọn ĐÚNG MỘT intent phù hợp với câu hỏi. CHỈ chọn trong danh sách.",
            "",
            "Danh sách intent:",
        ]
        for i in self._intents:
            ex = i.examples[0] if i.examples else ""
            lines.append(f"- {i.id}: {i.name}. Example: {ex}")
        if self.few_shot_examples:
            lines.append("")
            lines.append("Ví dụ:")
            for q, tid in self.few_shot_examples[:10]:
                lines.append(f"  Q: {q} → {tid}")
        lines.append("")
        lines.append(f"Câu hỏi: {query}")
        lines.append("Intent:")
        return "\n".join(lines)

    def _parse_intent(self, text: str) -> str | None:
        """Parse intent_id từ output."""
        text = text.strip().lower()
        for i in self._intents:
            if i.id.lower() in text or i.name.lower() in text:
                return i.id
        # Tìm pattern "intent_id" hoặc dòng đầu
        first = text.split("\n")[0].strip()
        for i in self._intents:
            if i.id.lower() == first or i.id.lower() in first:
                return i.id
        return None
