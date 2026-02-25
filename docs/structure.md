### **1. Mục đích & Bài toán giải quyết**

Mục đích chính của REIC là cải thiện intent classification trong customer service e-commerce, nơi intent (ý định người dùng như "track order" hay "change address") cần được phân loại chính xác để routing đến agent phù hợp, giảm thời gian xử lý và chi phí. Bài báo tập trung vào các vấn đề scalability khi số intent tăng (hàng trăm đến nghìn) do mở rộng sản phẩm và sự khác biệt taxonomy giữa các verticals (e.g., third-party vs first-party products tại Amazon).

Các phương pháp truyền thống (bag-of-words, SVM, LSTM) hoặc hiện đại (transformers như BERT/XLM, retrieval-based few-shot như Yu et al. 2021 hoặc Liu et al. 2024) chủ yếu xử lý intent ở quy mô nhỏ hoặc few-shot, bỏ qua: (1) Scalability với intent lớn và động (phải retrain khi thêm intent mới, softmax lớn dẫn đến tốn kém compute); (2) Label correlation trong hierarchical taxonomy (flat classification忽略 quan hệ cha-con, dễ nhầm lẫn intent gần nghĩa); (3) Dynamic intent space (intent thay đổi thường xuyên theo business, nhưng mô hình cũ không linh hoạt); (4) Hallucination của LLM (zero/few-shot LLMs có thể generate intent không tồn tại); (5) Áp dụng thực tế ở multi-domain/large-scale (hầu hết nghiên cứu tập trung cross-domain nhỏ, không xử lý out-of-domain intents tốt). REIC lấp gap bằng cách kết hợp RAG với hierarchical ontology, cho phép cập nhật intent mà không retrain, và constrained reasoning để tránh hallucination.

**Thứ 1: Intent classification truyền thống là flat multiclass classification**

- **Khi số intent k lớn (hàng trăm–hàng nghìn):**
    - Softmax rất lớn
    - Training/inference tốn kém
    - Mỗi intent mới → phải retrain lại mô hình

**⇒ Cơ chế của REIC:**

- Không phân loại trên toàn bộ tập intent T
- Thay vào đó:
    1. **Retrieve** một tập nhỏ intent candidates E⊂TE
    2. **LLM chỉ xếp hạng (rerank)** trong tập E

## **Thứ 2: Các mô hình trước đã bỏ qua correlation của các label**

- Bỏ qua:
    - Quan hệ cha–con
    - Quan hệ phân cấp (hierarchy)
- Dẫn đến nhầm lẫn giữa các intent gần nghĩa

Flat model **không biết**:

- “Change Shipping Address” là **một sub-intent của Shipping**

**Cơ chế của REIC:**

- Intent được tổ chức theo **hierarchical ontology**
- Retrieval có vai trò:
    - Chỉ retrieve các intent **nằm cùng nhánh hierarchy**
- LLM chỉ cần phân biệt giữa các intent **có quan hệ ngữ nghĩa gần,** giảm nhầm lẫn giữa các sub-intents.

B. REIC – hierarchical + retrieval

**Bước 1: Hierarchical ontology**

```
Level1:Order-related
Level2:Shipping
Level3:
-ChangeShippingAddress
-DeliveryStatus

```

---

**Bước 2: Retrieval chọn đúng nhánh**

Query được encode → retrieval trả về:

```
Candidate intents:
1. Change Shipping Address
2. Delivery Status

```

**Bước 3: LLM reranking**

LLM chỉ cần phân biệt:

- “Change Shipping Address”
- “Delivery Status”

## Thứ 3: Dynamic intent space – Intent thay đổi liên tục

### Vấn đề

- Trong hệ thống thực:
    - Intent mới xuất hiện thường xuyên do nhu cầu người dùng liên tục cập nhật
    - Taxonomy thay đổi theo business
- Các mô hình phân loại trước đây :
    - Không linh hoạt do phải retrain lại mỗi khi có một intent mới

### ⇒ REIC giải quyết như thế nào?

**Cơ chế của REIC:**

| Thành phần | Vai trò |
| --- | --- |
| **Knowledge** (**Index)(retriever)** | Lưu & cập nhật |
| **Reasoning** (suy luận chọn intent -LLM ) | Phán đoán |

### Knowledge ở đây là gì?

Không phải kiến thức chung, mà là:

- Các **ví dụ truy vấn – intent:** Mô tả intent và các intent mới được thêm

Ví dụ index chứa:

```
("I wantto change myaddress", Change ShippingAddress)
("Where is my package?", TrackOrder)
("I wantto return this item", ReturnOrder)

```

### Index làm gì?

- Lưu các ví dụ này dưới dạng **vector**
- Khi có query mới: tìm các ví dụ **gần nhất về ngữ nghĩa**

**Quan trọng:**

- **Index có thể cập nhật liên tục**
- Thêm intent mới → chỉ cần thêm dữ liệu vào index, **không cần retrain LLM**

### Reasoning là gì?

👉 Là quá trình:

- Đọc câu query và so sánh với các intent candidates để đưa ra quyết định intent đúng nhất

📌 REIC **không dùng LLM để “nhớ” toàn bộ intent**

LLM chỉ:

- Nhận **context ngắn** (top-k intent candidates)
- Dùng năng lực **ngôn ngữ & suy luận** để chọn

## Thứ  4: Hallucination & reliability của LLM

### Vấn đề

- LLM khi generate:
    - Có thể sinh intent không tồn tại (halluciation)

### REIC giải quyết như thế nào?

**Cơ chế của REIC:**

- Không để LLM generate tự do
- Áp dụng:
    - **Constrained decoding**
    - **Probability-based intent scoring**
- LLM:
    - Chỉ đánh giá xác suất cho các intent có sẵn trong candidate list

**Kết quả:**

- Không sinh intent ngoài tập cho phép

---

### 🧠 **2. Cách hoạt động / Kiến trúc tổng quan**

**Cách hoạt động / Kiến trúc tổng quan**
REIC hoạt động theo pipeline RAG-style: Retriever + Generator (LLM). Kiến trúc bao gồm:

1. **Hierarchical Ontology**: Intent được cấu trúc cây (e.g., Figure 1 trong bài: Vertical1 third-party có 3 levels coarse-to-fine; Vertical2 first-party rộng hơn với device-specific intents).
2. **Retriever (Knowledge Index)**: Sử dụng dense retriever (e.g., DPR - Dense Passage Retriever) để encode query q và retrieve top-k intent candidates từ index vector chứa intent descriptions + examples (e.g., ("Where is my package?", TrackOrder)). Index động, cập nhật dễ dàng.
3. **Reasoning (LLM Reranker)**: LLM (e.g., GPT-4 hoặc fine-tuned LLaMA) nhận prompt với query + candidates, sử dụng in-context learning (ICL) để score và chọn intent tốt nhất (argmax P(t|q; E)). Constrained để tránh hallucination.
Ví dụ đơn giản: Query "I want to change my address" → Retriever lấy candidates ["ChangeShippingAddress", "DeliveryStatus"] từ nhánh Shipping → LLM rerank và chọn "ChangeShippingAddress".
Sơ đồ (từ Figure 2 trong bài): Query → Encoder → Retrieve from Index → Candidates → LLM Prompt (ICL) → Predicted Intent.