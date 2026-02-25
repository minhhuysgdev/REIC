"""
RoBERTa Baseline - Fine-tuned với classification head (paper Section 5.2).
RoBERTa-base + linear layer → num_labels (flat multiclass).
"""

import math
from pathlib import Path

from reic.models import IntentDefinition


class RoBERTaClassifier:
    """
    Fine-tuned RoBERTa với multiple classification heads.
    Pooled embedding → linear(num_labels) → softmax.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        checkpoint_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None
        self._tokenizer = None
        self._id2idx: dict[str, int] = {}
        self._idx2intent: list[IntentDefinition] = []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("Cần: pip install transformers torch") from e

        load_path = str(self.checkpoint_path) if self.checkpoint_path and self.checkpoint_path.exists() else self.model_name
        num_labels = len(self._idx2intent) if self._idx2intent else 8  # default

        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path if (self.checkpoint_path and self.checkpoint_path.exists()) else self.model_name
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            load_path if (self.checkpoint_path and self.checkpoint_path.exists()) else self.model_name,
            num_labels=num_labels,
        )
        self._model.eval()
        self._torch = torch

    def fit(self, intents: list[IntentDefinition]) -> None:
        """Đăng ký danh sách intent (cần trùng với lúc train)."""
        self._idx2intent = [i for i in intents if i.examples]
        self._id2idx = {i.id: idx for idx, i in enumerate(self._idx2intent)}

    def predict(
        self, query: str, top_k: int = 5
    ) -> tuple[str, str, float, dict[str, float]]:
        """
        Dự đoán: logits → softmax → P(tj|q).
        """
        if not self._idx2intent:
            return "", "", 0.0, {}

        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        probs = self._softmax(logits.tolist())
        P_tj = {self._idx2intent[i].id: probs[i] for i in range(len(self._idx2intent))}
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best = self._idx2intent[best_idx]
        return best.id, best.name, probs[best_idx], P_tj

    def _softmax(self, x: list[float]) -> list[float]:
        m = max(x)
        exp_x = [math.exp(v - m) for v in x]
        s = sum(exp_x)
        return [e / s for e in exp_x]
