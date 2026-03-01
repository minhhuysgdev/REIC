"""
REIC Pipeline: Query → Encoder → Index Search → Candidates → Rerank → Predicted Intent
Theo Figure 2: Customer query → Query Encoder → Index Search → Retrieved intent candidates
→ Probability calculation → Intent LLM → Predicted Intent → Routing
"""

from pathlib import Path

from reic.models import ReicResult
from reic.index import KnowledgeIndex
from reic.ontology import IntentOntology
from reic.reranker import (
    LLMReranker,
    LocalLLMReranker,
    Reranker,
    SimilarityReranker,
)
from reic.retriever import Retriever


class ReicPipeline:
    """
    Pipeline REIC end-to-end:
    1. Query Encoder + Index Search (retrieve top-k candidates)
    2. Rerank (constrained: chỉ chọn trong candidates)
    """

    def __init__(
        self,
        ontology_path: str | Path,
        top_k: int = 5,
        use_llm: bool = False,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
        adapter_path: str | Path | None = None,
        backend: str = "tfidf",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.ontology = IntentOntology.from_json(ontology_path)
        self.index = KnowledgeIndex(
            self.ontology, backend=backend, model_name=model_name
        )
        self.retriever = Retriever(self.index, top_k=top_k)
        if use_local_llm:
            self.reranker: Reranker = LocalLLMReranker(
                model_name=local_llm_model,
                adapter_path=str(adapter_path) if adapter_path else None,
            )
        elif use_llm:
            self.reranker = LLMReranker()
        else:
            self.reranker = SimilarityReranker()

    def predict(self, query: str) -> ReicResult:
        """
        Dự đoán intent từ customer query.
        """
        candidates = self.retriever.retrieve(query)
        if not candidates:
            return ReicResult(
                predicted_intent="",
                intent_id="",
                path=[],
                confidence=0.0,
                candidates=[],
                intent_probabilities={},
                vertical="3P",
            )
        intent_id, intent_name, confidence, P_tj = self.reranker.rerank(query, candidates)
        intent_def = self.ontology.get_intent(intent_id)
        vertical = intent_def.vertical if intent_def else "3P"
        path = intent_def.path if intent_def else []
        return ReicResult(
            predicted_intent=intent_name,
            intent_id=intent_id,
            path=path,
            confidence=confidence,
            candidates=candidates,
            intent_probabilities=P_tj,
            vertical=vertical,
        )
