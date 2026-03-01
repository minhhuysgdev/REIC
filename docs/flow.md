# Luồng đi (Flow) – Project REIC

Tài liệu mô tả luồng xử lý và các model trong project REIC.

---

## 1. Luồng tổng thể (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Query (câu hỏi khách hàng)                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Ontology (data/ontology.json)                                                   │
│  Intent phân cấp 3P/1P, mỗi intent: id, name, description, path, examples       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Knowledge Index (TF-IDF hoặc Dense)                                              │
│  Mỗi (intent + example) = 1 document → encode → lưu vector                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Retriever                                                                       │
│  Encode query → search index → trả về top-k IntentCandidate                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Reranker (Similarity / OpenAI LLM / Local LLM)                                  │
│  Tính P(tj|q,E) → argmax → intent dự đoán + confidence                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReicResult                                                                      │
│  predicted_intent, intent_id, path, confidence, candidates, intent_probabilities │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Code:** `reic/pipeline.py` → `ReicPipeline.predict(query)`.

---

## 2. Chi tiết từng bước

### 2.1 Ontology

| Thành phần | File | Mô tả |
|------------|------|--------|
| `IntentOntology` | `reic/ontology.py` | Load từ `data/ontology.json`. Intent có `path` (vd. `["Order-related", "Shipping", "ChangeShippingAddress"]`), `vertical` (3P/1P). |
| `iter_leaf_intents()` | — | Chỉ lấy intent lá (có `examples`) để đưa vào index. |

### 2.2 Index (Retrieval)

| Backend | Class | Model / Cách hoạt động |
|---------|--------|------------------------|
| **tfidf** | `TfidfIndex` | Không dùng model. Tokenize → TF-IDF → cosine similarity. Nhanh, demo. |
| **dense** | `DenseIndex` | `SentenceTransformer` (mặc định `paraphrase-multilingual-MiniLM-L12-v2`). Encode document + query → cosine. |

**Factory:** `KnowledgeIndex(ontology, backend="tfidf"|"dense", model_name=...)` → `reic/index.py`.

### 2.3 Retriever

- **Input:** `query: str`
- **Output:** `list[IntentCandidate]` (top-k), mỗi candidate có `intent_id`, `name`, `description`, `path`, `example`, `score`
- **File:** `reic/retriever.py` → `Retriever.retrieve(query)`

### 2.4 Reranker

| Loại | Class | Cách tính P(tj\|q,E) |
|------|--------|----------------------|
| **Similarity** | `SimilarityReranker` | Softmax(scores từ retriever). Không gọi LLM. |
| **OpenAI** | `LLMReranker` | Gọi API (vd. gpt-4o-mini), prompt trả JSON xác suất → parse → chuẩn hóa. |
| **Local LLM** | `LocalLLMReranker` | Qwen2-1.5B (hoặc model HF ~1.5B). Constrained decoding: với mỗi intent t_j, forward P(q,E)+t_j → log-prob → P(tj\|q,E). Có thể gắn LoRA qua `adapter_path`. |

**Output reranker:** `(intent_id, intent_name, confidence, P_tj)` với `P_tj = {intent_id: probability}`.

### 2.5 Kết quả

`ReicResult` (`reic/models.py`):

- `predicted_intent`, `intent_id`, `path`, `vertical`
- `confidence` = P(t̂|q,E)
- `candidates` = top-k từ retriever
- `intent_probabilities` = P(tj|q,E) cho mọi candidate

---

## 3. Luồng theo từng chế độ chạy

### 3.1 REIC: tfidf + Similarity

```
Query → TfidfIndex.search() → top-k candidates → SimilarityReranker (softmax) → ReicResult
```

- Dùng khi: demo nhanh, không cần model, không API.

### 3.2 REIC: dense + Similarity

```
Query → DenseIndex (SentenceTransformer encode) → top-k → SimilarityReranker → ReicResult
```

- Dùng khi: muốn retrieval tốt hơn, vẫn không gọi LLM.

### 3.3 REIC: dense + OpenAI LLM

```
Query → DenseIndex → top-k → LLMReranker (gpt-4o-mini) → parse JSON P(tj|q,E) → argmax → ReicResult
```

- Cần `OPENAI_API_KEY`. Đúng hướng paper: LLM chỉ chọn trong E (constrained).

### 3.4 REIC: dense + Local LLM (Qwen)

```
Query → DenseIndex → top-k → LocalLLMReranker (Qwen2-1.5B, optional LoRA) → P(tj|q,E) từ log-prob → ReicResult
```

- Chạy local, có thể fine-tune (LoRA) qua `adapter_path`.

---

## 4. Baselines (so sánh, không nằm trong pipeline REIC)

| Baseline | Model | Cách dùng |
|----------|--------|-----------|
| **BertClassifier** | Cross-encoder (ms-marco-MiniLM-L-6-v2) | [query, intent] → score → chọn intent điểm cao nhất. |
| **RoBERTaClassifier** | RoBERTa + classification head | Fine-tune trên intent labels; load checkpoint. |
| **QwenClassifier** | Qwen2-1.5B-Instruct | Zero-shot: prompt + danh sách intent → generate 1 intent. |
| **LLMClassifier** | Qwen2-1.5B + classification head | Fine-tune head (có thể + LoRA); load checkpoint. |

So sánh REIC vs baselines: `benchmarks/compare.py`, notebook `notebooks/finetune_and_compare.ipynb`.

---

## 5. Sơ đồ file tham chiếu

```
data/ontology.json          → IntentOntology
reic/ontology.py            → IntentOntology
reic/index.py               → TfidfIndex, DenseIndex, KnowledgeIndex
reic/retriever.py           → Retriever
reic/reranker.py            → SimilarityReranker, LLMReranker, LocalLLMReranker
reic/pipeline.py            → ReicPipeline
reic/models.py              → IntentDefinition, IntentCandidate, ReicResult
reic/baselines/             → BertClassifier, RoBERTaClassifier, QwenClassifier, LLMClassifier
```

---

## 6. Công thức (theo paper)

- **Retrieval:** Lấy tập E ⊂ T (top-k intent candidates từ index).
- **Rerank:** P(tj|q,E) = model(q, E) cho từng t_j ∈ E; **t̂ = argmax P(tj|q,E)**.
- **Constrained:** Chỉ được chọn intent trong E, tránh hallucination.
