# So sánh REIC với các baseline (Section 5.2)

## Các phương pháp trong paper

| Method | Mô tả |
|--------|-------|
| **RoBERTa** | Fine-tune RoBERTa-base với classification heads |
| **Mistral 7B** | Fine-tune Mistral 7B với sequence classification head |
| **Claude Zero-shot** | Claude 3.5 Sonnet, prompt định nghĩa toàn bộ intent |
| **Claude Few-shot** | Claude + 20 examples (10/vertical) |
| **Claude+RAG** | Claude + retrieved candidates (giống REIC) |
| **REIC** | Retrieval + fine-tuned LLM reranker |

## Baselines thêm vào (model nhỏ)

| Method | Mô tả | Kích thước |
|--------|-------|------------|
| **BERT** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) score (query, intent) | ~80M params |
| **Qwen 1.5B** | Qwen2-1.5B-Instruct zero-shot/few-shot | 1.5B params |

## So sánh nhanh

| Tiêu chí | REIC | BERT | Qwen 1.5B |
|----------|------|------|-----------|
| **Scalability** | ✓ Chỉ rerank top-k | ✗ Score toàn bộ intent | ✗ Prompt toàn bộ intent |
| **Dynamic intent** | ✓ Thêm vào index | ✗ Cần retrain/refit | △ Thêm vào prompt |
| **Hierarchy** | ✓ Retrieve theo nhánh | ✗ Flat | ✗ Flat |
| **Hallucination** | ✓ Constrained | ✓ Constrained | △ Có thể sinh intent lạ |
| **Latency** | Thấp (retrieve + rerank) | Trung bình (N pairs) | Cao (generate) |
| **VRAM** | Thấp | ~500MB | ~3GB |

## Chạy benchmark

```bash
# Cài đủ deps
pip install sentence-transformers transformers torch accelerate

# Chạy so sánh
python benchmarks/compare.py
```

## Kết luận

- **REIC** phù hợp khi: intent lớn, thay đổi thường xuyên, cần hierarchy.
- **BERT** phù hợp khi: intent cố định, ít class, cần latency thấp.
- **Qwen 1.5B** phù hợp khi: cần zero-shot linh hoạt, chấp nhận latency cao.
