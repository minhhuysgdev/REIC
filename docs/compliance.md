# Đối chiếu implementation với phương pháp REIC (paper)

So sánh pipeline hiện tại với mô tả: Index (dense, hierarchical) → Candidate retrieval (top-k, cosine) → Intent probability (LLM constrained decoding) → argmax.

---

## 1. Index construction (xây dựng index vector)

| Yêu cầu paper | Hiện tại | Tuân thủ? |
|---------------|----------|-----------|
| Dataset annotated (query, intent) — e.g. 52k samples | Index build từ **ontology**: mỗi doc = (name + description + example) per intent; không tách riêng file (query, intent) hàng loạt | **Một phần**: có cặp (text, intent) qua examples; chưa hỗ trợ load 52k cặp (query, intent) riêng |
| Encode query → dense vector (e.g. MPNet) | **Dense**: SentenceTransformer (mặc định MiniLM). **TF-IDF**: không dense | **Đúng** (dense): encoder + vector; có thể đổi model (MPNet) qua `model_name` |
| Lưu hierarchical (Vertical > Category > Sub-issue) | Ontology có `path` và `vertical`; index lưu (doc, intent) với intent có `.path` | **Đúng**: cấu trúc phân cấp có trong ontology |
| Dense vector store + **ANN search** | Dense: brute-force cosine (toàn bộ vectors). Không dùng FAISS/ANN | **Chưa**: đúng dense, thiếu ANN (nên thêm khi scale ~10³ intents) |

---

## 2. Candidate retrieval (tìm top-k ứng viên)

| Yêu cầu paper | Hiện tại | Tuân thủ? |
|---------------|----------|-----------|
| Encode query → \(v_q\), cùng encoder | DenseIndex: `model.encode([query])` | **Đúng** |
| Top-k (e.g. k=10) bằng cosine similarity \(\frac{v_q \cdot v_i}{\|v_q\|\|v_i\|}\) | `scores = dot(embeddings, q_emb) / (norm(...))`; `top_k` cấu hình được (mặc định 5) | **Đúng** |
| Set E = k **unique** intents | Dedup theo `intent.id` trong `search()` | **Đúng** |

---

## 3. Hierarchical classification

| Yêu cầu paper | Hiện tại | Tuân thủ? |
|---------------|----------|-----------|
| Phân loại coarse → fine; mỗi head \<50 intents | Ontology có path (Level1 → 2 → 3); **retrieval hiện tại là flat** — không “chọn nhánh trước, rồi search trong nhánh” | **Chưa**: chưa retrieval theo hierarchy (ví dụ: retrieve theo vertical/category trước, rồi mới top-k leaf trong nhánh đó) |

---

## 4. Intent probability & reranking (LLM)

| Yêu cầu paper | Hiện tại | Tuân thủ? |
|---------------|----------|-----------|
| LLM **fine-tuned** (e.g. Mistral-7B + LoRA) | **LLMReranker**: OpenAI API (gpt-4o-mini), không fine-tune. **SimilarityReranker**: softmax(similarity), không LLM | **Chưa**: chưa có LLM fine-tuned tại chỗ (Mistral + PEFT/LoRA) |
| Prompt \(\mathcal{P}\): instructions + query \(q\) + set E | LLMReranker: prompt có query + danh sách intent (E) + example | **Đúng** |
| **Constrained decoding**: không auto-regressive; batch forward, append \(t_j\), mask prompt, log-prob cho \(t_j\) | Yêu cầu LLM output JSON probability rồi parse; **không** dùng logits/log-prob trực tiếp từ model | **Chưa**: chưa constrained decoding với log-prob thật từ forward pass |
| \(P(t_j \mid q, E)\) = normalized log-prob, \(\hat{t} = \arg\max_{t_j \in E} P(t_j \mid q, E)\) | Output luôn trong E; chọn argmax P_tj (từ softmax hoặc JSON parse) | **Đúng** (ý nghĩa: chỉ chọn trong E, argmax) |
| Tránh hallucination (chỉ intent trong E) | Reranker chỉ nhận candidates từ E; không generate intent mới | **Đúng** |

---

## 5. Dynamic update & trade-offs

| Yêu cầu paper | Hiện tại | Tuân thủ? |
|---------------|----------|-----------|
| Thêm (query, intent) vào index, không retrain LLM | Có thể thêm intent vào ontology và rebuild index; `DenseIndex` có thể mở rộng (chưa có API `add (query, intent)` trực tiếp từ file 52k) | **Một phần**: thêm intent → rebuild index được; chưa API “append (query, intent)” cho dataset lớn |
| Top-k cân bằng accuracy vs latency | `top_k` cấu hình được trong pipeline | **Đúng** |

---

## Tóm tắt

| Thành phần | Trạng thái | Ghi chú |
|------------|------------|---------|
| Index: dense vector từ (query/intent-like text) | ✅ | Có dense + TF-IDF; có thể thêm tải 52k (query, intent) và dùng MPNet |
| Index: ANN search | ❌ | Đang brute-force; nên thêm FAISS/ANN khi scale lớn |
| Retrieval: top-k, cosine, unique E | ✅ | Đúng |
| Hierarchy trong ontology | ✅ | Có path/vertical |
| Retrieval theo hierarchy (nhánh rồi leaf) | ❌ | Đang flat; có thể thêm bước “chọn vertical/category → search trong nhánh” |
| Rerank: chỉ chọn trong E, argmax | ✅ | Constrained đúng nghĩa (không hallucinate intent mới) |
| Rerank: LLM fine-tuned + constrained decoding (log-prob) | ✅ (option) | **LocalLLMReranker**: model ~1.5B (Qwen2-1.5B), constrained decoding — forward P(q,E)+t_j → log-prob → P(tj\|q,E). Hỗ trợ `adapter_path` (LoRA/PEFT) để dùng model đã fine-tune. Pipeline: `use_local_llm=True`, `adapter_path=...` (optional). |

**Kết luận:** Pipeline hiện tại **tuân thủ đúng** phần: index vector (dense), retrieval top-k cosine, ontology phân cấp, constrained (chỉ E), argmax, dynamic update, **và** LLM reranker local với constrained decoding (log-prob) qua `LocalLLMReranker` (model ~1.5B, fine-tune được bằng LoRA). **Chưa tuân thủ đúng**: (1) retrieval theo hierarchy (flat thay vì coarse-to-fine), (2) ANN thay brute-force.
