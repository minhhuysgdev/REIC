#!/usr/bin/env python3
"""
Tách examples từ ontology.json thành 3 tập: train, dev, test (stratified).
Xuất data/train.csv, dev.csv, test.csv và data/train.json, dev.json, test.json.
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reic.ontology import IntentOntology


def load_examples_by_intent(ontology_path: Path) -> list[tuple[str, str]]:
    """(text, intent_id) từ mọi example trong ontology."""
    ontology = IntentOntology.from_json(ontology_path)
    data = []
    for intent in ontology.iter_leaf_intents():
        for ex in intent.examples:
            data.append((ex.strip(), intent.id))
    return data


def stratified_split(
    data: list[tuple[str, str]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Chia theo intent (stratified).
    Mỗi intent: ít nhất 1 dev nếu có >= 2 mẫu, ít nhất 1 test nếu có >= 3 mẫu, còn lại train.
    """
    random.seed(seed)
    by_intent: dict[str, list[str]] = {}
    for text, intent_id in data:
        by_intent.setdefault(intent_id, []).append(text)

    train, dev, test = [], [], []
    for intent_id, texts in by_intent.items():
        random.shuffle(texts)
        n = len(texts)
        n_dev = 1 if n >= 2 else 0
        n_test = 1 if n >= 3 else 0
        n_train = n - n_dev - n_test
        i = 0
        for _ in range(n_train):
            train.append((texts[i], intent_id))
            i += 1
        for _ in range(n_dev):
            dev.append((texts[i], intent_id))
            i += 1
        for _ in range(n_test):
            test.append((texts[i], intent_id))
            i += 1

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ontology_path = root / "data" / "ontology.json"
    if not ontology_path.exists():
        print(f"Không tìm thấy {ontology_path}")
        sys.exit(1)

    data = load_examples_by_intent(ontology_path)
    print(f"Tổng số mẫu từ ontology: {len(data)}")

    train, dev, test = stratified_split(data, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15)

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        path_csv = data_dir / f"{name}.csv"
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"  {name}.csv: {len(rows)} mẫu -> {path_csv}")

        path_json = data_dir / f"{name}.json"
        items = [{"text": text, "label": label} for text, label in rows]
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  {name}.json: {len(rows)} mẫu -> {path_json}")

    # Label list (thứ tự đồng bộ với train_baselines)
    labels = sorted(set(r[1] for r in train + dev + test))
    labels_path = data_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    print(f"  labels.txt: {len(labels)} intent -> {labels_path}")


if __name__ == "__main__":
    main()
