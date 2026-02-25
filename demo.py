#!/usr/bin/env python3
"""
Demo REIC - Intent Classification.
Chạy: python demo.py [query]
Hoặc: python demo.py  (interactive)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reic.pipeline import ReicPipeline


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    ontology_path = data_dir / "ontology.json"
    if not ontology_path.exists():
        print(f"Không tìm thấy {ontology_path}")
        sys.exit(1)

    print("Đang tải REIC...")
    pipeline = ReicPipeline(ontology_path, top_k=5, use_llm=False)
    print("Sẵn sàng. Nhập câu hỏi (hoặc Enter để thoát):\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        _run_query(pipeline, query)
    else:
        while True:
            try:
                query = input("> ").strip()
                if not query:
                    break
                _run_query(pipeline, query)
            except (EOFError, KeyboardInterrupt):
                break


def _run_query(pipeline: ReicPipeline, query: str) -> None:
    result = pipeline.predict(query)
    print(f"\nQuery: {query}")
    print(f"→ Intent: {result.predicted_intent} (P(t̂|q,E) = {result.confidence:.3f})")
    print(f"  Path: {' → '.join(result.path)}")
    print(f"  Vertical: {result.vertical}")
    if result.intent_probabilities:
        print("  P(tj|q,E):")
        for tid, p in sorted(result.intent_probabilities.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {tid}: {p:.3f}")
    if result.candidates:
        print("  Top candidates (retrieval):")
        for c in result.candidates[:3]:
            print(f"    - {c.name} (score: {c.score:.3f})")
    print()


if __name__ == "__main__":
    main()
