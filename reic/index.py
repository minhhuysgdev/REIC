"""
Knowledge Index - lưu (query example, intent) dưới dạng vector.
Retrieve top-k intent candidates theo similarity.
Hỗ trợ: TF-IDF (nhanh, không tải model) hoặc SentenceTransformer (dense).
"""

import re
from typing import Literal

import numpy as np

from reic.models import IntentDefinition
from reic.ontology import IntentOntology


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho TF-IDF."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


class TfidfIndex:
    """
    TF-IDF index - chạy ngay, không cần tải model.
    Phù hợp demo nhanh.
    """

    def __init__(self, ontology: IntentOntology):
        self.ontology = ontology
        self._documents: list[tuple[str, IntentDefinition, str]] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._build_index()

    def _build_index(self) -> None:
        self._documents = []
        for intent in self.ontology.iter_leaf_intents():
            text_base = f"{intent.name} {intent.description}"
            for ex in intent.examples:
                doc_text = f"{text_base} Example {ex}"
                self._documents.append((doc_text, intent, ex))
        if not self._documents:
            return
        # Build vocab
        doc_tokens = [_tokenize(d[0]) for d in self._documents]
        vocab = {}
        for tokens in doc_tokens:
            for t in tokens:
                vocab.setdefault(t, len(vocab))
        self._vocab = vocab
        # TF
        n_docs = len(doc_tokens)
        n_vocab = len(vocab)
        tf = np.zeros((n_docs, n_vocab))
        for i, tokens in enumerate(doc_tokens):
            for t in tokens:
                if t in vocab:
                    tf[i, vocab[t]] += 1
        # IDF
        df = (tf > 0).sum(axis=0)
        self._idf = np.log((n_docs + 1) / (df + 1)) + 1
        self._matrix = tf * self._idf
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._matrix = self._matrix / norms

    def search(self, query: str, top_k: int = 5) -> list[tuple[IntentDefinition, str, float]]:
        if not self._documents or self._matrix is None:
            return []
        tokens = _tokenize(query)
        q_vec = np.zeros(len(self._vocab))
        for t in tokens:
            if t in self._vocab:
                q_vec[self._vocab[t]] += 1
        q_vec = q_vec * self._idf
        norm = np.linalg.norm(q_vec)
        if norm < 1e-9:
            return []
        q_vec = q_vec / norm
        scores = self._matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][: top_k * 2]
        results = []
        seen = set()
        for idx in top_indices:
            _, intent, example = self._documents[idx]
            if intent.id not in seen and scores[idx] > 0:
                seen.add(intent.id)
                results.append((intent, example, float(scores[idx])))
                if len(results) >= top_k:
                    break
        return results


class DenseIndex:
    """Dense index dùng SentenceTransformer (tải model lần đầu)."""

    def __init__(
        self,
        ontology: IntentOntology,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        from sentence_transformers import SentenceTransformer

        self.ontology = ontology
        self.model = SentenceTransformer(model_name)
        self._documents: list[tuple[str, IntentDefinition, str]] = []
        self._embeddings: np.ndarray | None = None
        self._build_index()

    def _build_index(self) -> None:
        self._documents = []
        for intent in self.ontology.iter_leaf_intents():
            text_base = f"{intent.name}. {intent.description}"
            for ex in intent.examples:
                doc_text = f"{text_base}. Example: {ex}"
                self._documents.append((doc_text, intent, ex))
        if not self._documents:
            return
        texts = [d[0] for d in self._documents]
        self._embeddings = self.model.encode(texts, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[IntentDefinition, str, float]]:
        if not self._documents or self._embeddings is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, q_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        seen_ids = set()
        for idx in top_indices:
            _, intent, example = self._documents[idx]
            if intent.id not in seen_ids:
                seen_ids.add(intent.id)
                results.append((intent, example, float(scores[idx])))
        return results


def KnowledgeIndex(
    ontology: IntentOntology,
    backend: Literal["tfidf", "dense"] = "tfidf",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> TfidfIndex | DenseIndex:
    """Factory: tfidf (mặc định, nhanh) hoặc dense (chất lượng cao hơn)."""
    if backend == "dense":
        return DenseIndex(ontology, model_name)
    return TfidfIndex(ontology)
