"""
REIC - Retrieval-Enhanced Intent Classification
Pipeline: Query → Encoder → Index Search → Candidates → LLM Rerank → Predicted Intent
"""

from reic.pipeline import ReicPipeline
from reic.models import IntentCandidate, ReicResult

__all__ = ["ReicPipeline", "IntentCandidate", "ReicResult"]
