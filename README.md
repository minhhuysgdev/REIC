# REIC - Retrieval-Enhanced Intent Classification

Demo Python cho mô hình REIC (Retrieval-Enhanced Intent Classification) theo paper và cấu trúc trong `docs/structure.md`.

## Pipeline (Figure 2)

```
Customer query → Query Encoder → Index Search → Retrieved intent candidates
→ Probability calculation → Intent LLM → Predicted Intent → Routing
```

- **Retriever**: Dense vector index (sentence-transformers) chứa intent descriptions + examples
- **Reranker**: Chọn intent tốt nhất trong candidates (constrained, tránh hallucination)
- **Hierarchical ontology**: 3P (third-party) và 1P (first-party) verticals

## Cài đặt

**Dùng uv (khuyến nghị):**

```bash
# Tạo virtual env
uv venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Cài dependencies
uv pip install -r requirements.txt
```

Hoặc dùng pip:

```bash
pip install -r requirements.txt
```

## Chạy demo

```bash
# Interactive
python demo.py

# Hoặc truyền query trực tiếp
python demo.py "Update my shipping address"
python demo.py "Tôi muốn trả hàng"
python demo.py "Máy Kindle bị treo màn hình"
```

## Biến môi trường (optional)

| Biến | Mục đích |
|------|----------|
| `HF_TOKEN` | Hugging Face: tải model nhanh hơn, rate limit cao hơn (tránh warning). Lấy token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENAI_API_KEY` | Dùng LLM reranker (OpenAI) trong pipeline |

**Cách cấu hình:**

```bash
# Cách 1: export trong terminal (phiên hiện tại)
export HF_TOKEN=hf_xxxx

# Cách 2: file .env (không bị commit)
cp .env.example .env
# Sửa .env, thêm: HF_TOKEN=hf_xxxx
```

Sau đó load `.env` trong code (hoặc dùng `python-dotenv`). Jupyter/notebook: có thể chạy trong cell đầu `%env HF_TOKEN=hf_xxxx` hoặc export trước khi mở Jupyter.

## Dùng LLM reranker (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
# Sửa demo.py: ReicPipeline(..., use_llm=True)
```

## Cấu trúc

```
reic/
  __init__.py
  models.py      # IntentDefinition, IntentCandidate, ReicResult
  ontology.py    # Hierarchical ontology (3P/1P)
  index.py       # Knowledge Index (vector embeddings)
  retriever.py   # Query Encoder + Index Search
  reranker.py    # Similarity + LLM reranker (constrained)
  pipeline.py    # Full REIC pipeline
data/
  ontology.json  # Intent definitions + examples
demo.py
```

## Fine-tune BERT & Qwen → So sánh với REIC

Notebook `notebooks/finetune_and_compare.ipynb`:

1. **Setup & Load data** – ontology → training (query, intent_id)
2. **Fine-tune BERT** – bert-base-uncased + classification head → `checkpoints/bert/`
3. **Fine-tune Qwen** – Qwen2-1.5B + sequence classification head → `checkpoints/qwen/`
4. **Load models** – REIC, BERT-FT, Qwen-FT
5. **So sánh** – accuracy và chi tiết từng query

Chạy: mở notebook, Run All. Bước 2 ~ vài phút, bước 3 cần tải Qwen ~3GB và có thể chạy lâu hơn.

## Thêm intent mới

Chỉnh `data/ontology.json` hoặc gọi `index.add_intent()` — không cần retrain LLM.
