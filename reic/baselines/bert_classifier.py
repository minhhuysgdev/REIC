"""
BERT Baseline - tương tự RoBERTa trong paper (Section 5.2).
Flat multiclass: score toàn bộ intent set, không retrieval.
Dùng Cross-Encoder (sentence-transformers) hoặc BERT encoder.
"""

import math

from reic.models import IntentDefinition


class BertClassifier:
    """
    BERT baseline: flat multiclass classification.
    - Cross-Encoder: score (query, intent) pairs → P(tj|q)
    - Dùng sentence-transformers CrossEncoder (BERT-based, pretrained NLI)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        model_name: cross-encoder (ms-marco, ...) hoặc "bert-base-uncased" cho encoder-only
        """
        self.model_name = model_name
        self._model = None
        self._intents: list[IntentDefinition] = []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError:
                raise ImportError("Cần: pip install sentence-transformers") from None
        self._model = CrossEncoder(self.model_name)

    def fit(self, intents: list[IntentDefinition]) -> None:
        """Đăng ký danh sách intent (flat, không hierarchy)."""
        self._intents = [i for i in intents if i.examples]

    def predict(
        self, query: str, top_k: int = 5
    ) -> tuple[str, str, float, dict[str, float]]:
        """
        Dự đoán intent: score mỗi (query, intent) bằng Cross-Encoder.
        Trả về (intent_id, intent_name, confidence, P(tj|q)).
        """
        if not self._intents:
            return "", "", 0.0, {}

        self._ensure_loaded()
        pairs = []
        for i in self._intents:
            text = f"{i.name}. {i.description}. Example: {i.examples[0] if i.examples else ''}"
            pairs.append([query, text])

        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, float):
            scores = [scores]

        # Softmax → P(tj|q)
        x = [float(s) for s in scores]
        m = max(x)
        exp_x = [math.exp(v - m) for v in x]
        s = sum(exp_x)
        probs = [e / s for e in exp_x]
        P_tj = {self._intents[i].id: probs[i] for i in range(len(self._intents))}

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best = self._intents[best_idx]
        return best.id, best.name, probs[best_idx], P_tj
