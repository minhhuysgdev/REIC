"""
LLM Classification - Fine-tuned với sequence classification head (paper Section 5.2).
Mistral 7B: pooled embedding → linear(num_classes).
Demo: dùng model nhỏ hơn (Qwen 1.5B, Phi-2) vì Mistral 7B cần ~14GB VRAM.
"""

import math
from pathlib import Path

from reic.models import IntentDefinition


class LLMClassifier:
    """
    Fine-tuned LLM với sequence classification head.
    Thay vì generate, project pooled embedding → num_labels.
    Tương tự Mistral Classification trong paper.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        checkpoint_path: str | Path | None = None,
    ):
        """
        model_name: Mistral-7B, Qwen2-1.5B, ... (decoder-only)
        checkpoint_path: đường dẫn checkpoint đã fine-tune
        """
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None
        self._classifier_head = None
        self._tokenizer = None
        self._idx2intent: list[IntentDefinition] = []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("Cần: pip install transformers torch") from e

        load_path = str(self.checkpoint_path) if self.checkpoint_path and self.checkpoint_path.exists() else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        hidden_size = base.config.hidden_size
        num_labels = len(self._idx2intent) if self._idx2intent else 8

        # Classification head: pooled → num_labels
        self._model = base.model if hasattr(base, "model") else base.transformer
        self._classifier_head = nn.Linear(hidden_size, num_labels)

        if self.checkpoint_path and (self.checkpoint_path / "classifier.pt").exists():
            self._classifier_head.load_state_dict(
                torch.load(self.checkpoint_path / "classifier.pt", map_location="cpu")
            )

        self._model.eval()
        self._classifier_head.eval()
        self._torch = torch
        self._nn = nn

    def fit(self, intents: list[IntentDefinition]) -> None:
        """Đăng ký danh sách intent."""
        self._idx2intent = [i for i in intents if i.examples]

    def predict(
        self, query: str, top_k: int = 5
    ) -> tuple[str, str, float, dict[str, float]]:
        """
        Forward: lấy last hidden state → classifier head → softmax.
        """
        if not self._idx2intent:
            return "", "", 0.0, {}

        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Last token hidden state (hoặc mean pool)
            last_hidden = outputs.last_hidden_state[:, -1, :]  # (1, hidden_size)
            logits = self._classifier_head(last_hidden)[0].cpu().numpy()

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
