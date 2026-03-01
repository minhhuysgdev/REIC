#!/usr/bin/env python3
"""
Fine-tune RoBERTa và LLM với classification head.
Tạo training data từ ontology examples.
"""

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reic.ontology import IntentOntology


def load_train_data(ontology_path: Path) -> list[tuple[str, str]]:
    """(query, intent_id) từ ontology examples."""
    ontology = IntentOntology.from_json(ontology_path)
    data = []
    for intent in ontology.iter_leaf_intents():
        for ex in intent.examples:
            data.append((ex.strip(), intent.id))
    return data


def train_roberta(
    train_data: list[tuple[str, str]],
    output_dir: Path,
    model_name: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 8,
) -> None:
    """Fine-tune RoBERTa với AutoModelForSequenceClassification."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from torch.utils.data import Dataset
        import torch
    except ImportError as e:
        print(f"Cần: pip install transformers torch datasets - {e}")
        return

    # Build label map
    intents = sorted(set(t[1] for t in train_data))
    id2idx = {i: idx for idx, i in enumerate(intents)}
    num_labels = len(intents)

    class IntentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = [id2idx[l] for l in labels]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.encodings["input_ids"][i],
                "attention_mask": self.encodings["attention_mask"][i],
                "labels": torch.tensor(self.labels[i], dtype=torch.long),
            }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    texts = [t[0] for t in train_data]
    labels = [t[1] for t in train_data]
    dataset = IntentDataset(texts, labels, tokenizer)

    # Save label map
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"id2idx": id2idx, "idx2id": {v: k for k, v in id2idx.items()}}, f)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"RoBERTa saved to {output_dir}")


def train_llm(
    train_data: list[tuple[str, str]],
    output_dir: Path,
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    epochs: int = 2,
    batch_size: int = 2,
) -> None:
    """
    Fine-tune LLM với classification head.
    Freeze base, chỉ train linear head (nhanh, ít VRAM).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"Cần: pip install transformers torch - {e}")
        return

    intents = sorted(set(t[1] for t in train_data))
    id2idx = {i: idx for idx, i in enumerate(intents)}
    num_labels = len(intents)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Freeze base
    for p in base.parameters():
        p.requires_grad = False

    hidden_size = base.config.hidden_size
    model = base.model if hasattr(base, "model") else base.transformer
    classifier = nn.Linear(hidden_size, num_labels)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"id2idx": id2idx, "idx2id": {v: k for k, v in id2idx.items()}}, f)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            texts = [t[0] for t in batch]
            labels = [id2idx[t[1]] for t in batch]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            with torch.no_grad():
                out = model(**inputs)
            last_hidden = out.last_hidden_state[:, -1, :]
            logits = classifier(last_hidden)
            loss = nn.functional.cross_entropy(
                logits, torch.tensor(labels, dtype=torch.long)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_data):.4f}")

    torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    tokenizer.save_pretrained(str(output_dir))
    # Save config for inference
    base.save_pretrained(str(output_dir))
    print(f"LLM classifier saved to {output_dir}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ontology_path = root / "data" / "ontology.json"
    if not ontology_path.exists():
        print(f"Không tìm thấy {ontology_path}")
        sys.exit(1)

    train_data = load_train_data(ontology_path)
    print(f"Training samples: {len(train_data)}")

    out = root / "checkpoints"
    model = sys.argv[1] if len(sys.argv) > 1 else "roberta"

    if model == "roberta":
        train_roberta(train_data, out / "roberta", epochs=5)
    elif model == "llm":
        train_llm(train_data, out / "llm", epochs=3, batch_size=4)
    else:
        print("Usage: python train_baselines.py [roberta|llm]")


if __name__ == "__main__":
    main()
