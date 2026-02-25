#!/usr/bin/env python3
"""
So sánh REIC vs BERT vs Qwen 1.5B.
Tương tự Section 5.2 trong paper: RoBERTa, Mistral, Claude Zero/Few-shot, Claude+RAG.
"""

import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reic.ontology import IntentOntology
from reic.pipeline import ReicPipeline
from reic.baselines.bert_classifier import BertClassifier
from reic.baselines.roberta_classifier import RoBERTaClassifier
from reic.baselines.qwen_classifier import QwenClassifier


# Test queries (query, expected_intent_id)
TEST_QUERIES = [
    ("Update my shipping address", "change_shipping_address"),
    ("I want to change my address", "change_shipping_address"),
    ("Where is my package?", "delivery_status"),
    ("Tôi muốn trả hàng", "return_order"),
    ("I want to return this item", "return_order"),
    ("Leave at doorstep", "shipping_instructions"),
    ("Máy Kindle bị treo màn hình", "kindle_reset"),
    ("Alexa won't connect to wifi", "alexa_wifi"),
    ("Renew my streaming subscription", "subscription_renewal"),
]


def run_reic(pipeline: ReicPipeline, query: str) -> tuple[str, float]:
    r = pipeline.predict(query)
    return r.intent_id, r.confidence


def run_bert(clf: BertClassifier, query: str) -> tuple[str, float]:
    intent_id, _, conf, _ = clf.predict(query)
    return intent_id, conf


def run_qwen(clf: QwenClassifier, query: str) -> tuple[str, float]:
    intent_id, _, conf, _ = clf.predict(query)
    return intent_id, conf


def run_roberta(clf: RoBERTaClassifier, query: str) -> tuple[str, float]:
    intent_id, _, conf, _ = clf.predict(query)
    return intent_id, conf


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    ontology_path = data_dir / "ontology.json"
    if not ontology_path.exists():
        print(f"Không tìm thấy {ontology_path}")
        sys.exit(1)

    ontology = IntentOntology.from_json(ontology_path)
    intents = list(ontology.iter_leaf_intents())

    methods = {}
    results = {}

    # REIC (TF-IDF + Similarity reranker)
    print("Load REIC...")
    pipeline = ReicPipeline(ontology_path, top_k=5, backend="tfidf")
    methods["REIC"] = lambda q: run_reic(pipeline, q)

    # BERT (Cross-Encoder, không fine-tune)
    try:
        print("Load BERT (Cross-Encoder)...")
        bert = BertClassifier(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        bert.fit(intents)
        methods["BERT"] = lambda q: run_bert(bert, q)
    except Exception as e:
        print(f"  Skip BERT: {e}")
        methods["BERT"] = None

    # RoBERTa (fine-tuned với classification head)
    ckpt = Path(__file__).parent.parent / "checkpoints" / "roberta"
    try:
        print("Load RoBERTa (fine-tuned)...")
        roberta = RoBERTaClassifier(
            model_name="roberta-base",
            checkpoint_path=ckpt if ckpt.exists() else None,
        )
        roberta.fit(intents)
        methods["RoBERTa-FT"] = lambda q: run_roberta(roberta, q)
    except Exception as e:
        print(f"  Skip RoBERTa: {e}. Chạy: python scripts/train_baselines.py roberta")
        methods["RoBERTa-FT"] = None

    # Qwen 1.5B (bỏ qua nếu --quick)
    skip_qwen = "--quick" in sys.argv
    if not skip_qwen:
        try:
            print("Load Qwen 1.5B...")
            qwen = QwenClassifier(model_name="Qwen/Qwen2-1.5B-Instruct")
            qwen.fit(intents)
            methods["Qwen1.5B"] = lambda q: run_qwen(qwen, q)
        except Exception as e:
            print(f"  Skip Qwen: {e}")
            methods["Qwen1.5B"] = None
    else:
        print("Skip Qwen (--quick)")

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    for method_name, fn in methods.items():
        if fn is None:
            continue
        correct = 0
        total = len(TEST_QUERIES)
        times = []
        for query, expected in TEST_QUERIES:
            t0 = time.perf_counter()
            pred, conf = fn(query)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if pred == expected:
                correct += 1
        acc = correct / total
        avg_time = sum(times) / len(times) * 1000
        results[method_name] = {"accuracy": acc, "correct": correct, "total": total, "avg_ms": avg_time}
        print(f"{method_name:12} | Acc: {acc:.2%} ({correct}/{total}) | Avg: {avg_time:.0f} ms/query")

    print("=" * 60)
    print("\nChi tiết từng query:")
    print("-" * 60)
    for query, expected in TEST_QUERIES[:5]:
        print(f"\nQuery: {query}")
        print(f"  Expected: {expected}")
        for method_name, fn in methods.items():
            if fn is None:
                continue
            pred, conf = fn(query)
            ok = "✓" if pred == expected else "✗"
            print(f"  {method_name:12} → {pred} (conf: {conf:.2f}) {ok}")

    # Summary table
    print("\n" + "=" * 60)
    print("TÓM TẮT SO SÁNH (theo paper Section 5.2)")
    print("=" * 60)
    print("| Method     | Accuracy | Latency  | Ghi chú                    |")
    print("|------------|----------|----------|----------------------------|")
    for name, r in results.items():
        note = "Retrieval+Rerank" if name == "REIC" else "Flat/Zero-shot"
        print(f"| {name:10} | {r['accuracy']:>7.1%} | {r['avg_ms']:>6.0f}ms | {note:26} |")


if __name__ == "__main__":
    main()
