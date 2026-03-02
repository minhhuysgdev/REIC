#!/usr/bin/env python3
"""
Index construction: xây dựng Knowledge Index từ ontology.
- Đọc data/ontology.json
- Tạo documents: mỗi (intent + example) = 1 document
- Encode (TF-IDF hoặc Dense) và build index trong memory

Index được dùng bởi Retriever để lấy top-k intent candidates.
Có thể chạy độc lập để kiểm tra build thành công; hoặc index tự build khi tạo ReicPipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reic.ontology import IntentOntology
from reic.index import KnowledgeIndex


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ontology_path = root / "data" / "ontology.json"
    if not ontology_path.exists():
        print(f"Không tìm thấy {ontology_path}")
        sys.exit(1)

    backend = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    if backend not in ("tfidf", "dense"):
        print("Usage: python build_index.py [tfidf|dense]")
        sys.exit(1)

    print("Index construction")
    print("  Ontology:", ontology_path)
    print("  Backend:", backend)

    ontology = IntentOntology.from_json(ontology_path)
    intents = list(ontology.iter_leaf_intents())
    n_intents = len(intents)
    n_docs = sum(len(i.examples) for i in intents)

    print(f"  Leaf intents: {n_intents}")
    print(f"  Documents (intent+example): {n_docs}")

    index = KnowledgeIndex(
        ontology,
        backend=backend,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Sanity: search 1 query
    test_query = "I want to change my address"
    hits = index.search(test_query, top_k=3)
    print(f"\n  Test search: \"{test_query}\"")
    print(f"  Top-3: {[(h[0].name, h[2]) for h in hits]}")

    print("\nIndex construction xong. Index đang trong memory (chưa lưu disk).")
    print("Pipeline/Streamlit sẽ build index tương tự khi khởi tạo.")


if __name__ == "__main__":
    main()
