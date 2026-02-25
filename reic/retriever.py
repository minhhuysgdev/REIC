"""
Retriever: Query Encoder + Index Search.
Trả về top-k intent candidates (constrained trong hierarchy).
"""

from reic.index import KnowledgeIndex
from reic.models import IntentCandidate


class Retriever:
    """Query → encode → search index → intent candidates."""

    def __init__(self, index: KnowledgeIndex, top_k: int = 5):
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> list[IntentCandidate]:
        """
        Retrieve top-k intent candidates từ knowledge index.
        Chỉ trả về intents có trong ontology (constrained).
        """
        hits = self.index.search(query, top_k=self.top_k)
        return [
            IntentCandidate(
                intent_id=intent.id,
                name=intent.name,
                description=intent.description,
                path=intent.path,
                example=example,
                score=score,
            )
            for intent, example, score in hits
        ]
