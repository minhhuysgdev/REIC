### 1. Định nghĩa các Vertical (Nhánh kinh doanh)

Trong nghiên cứu này, tác giả chia dữ liệu ý định thành hai nhánh chính:

**3P Business (Third-Party):** Liên quan đến các đơn hàng bán lẻ từ bên thứ ba
Ví dụ: mô hình Sàn giap dịch, công ty TMĐT chỉ đóng vai trò là nơi dể các seller đăng bài

- **Ví dụ về intent:**
    - Shop X giao hàng bị thiếu, cần khiếu nại
    - Shop ơi size  phù hợp với người nặng bao nhiêu cân

**1P Business (First-Party):** Liên quan đến các thiết bị độc quyền (proprietary devices) và các dịch vụ kỹ thuật số của chính công ty cung cấp.

Công ty điện tử tự nhập hàng về kho, tự bán và tự chịu trách nhiệm về sản phẩm dưới thương hiệu của chính mình hoặc các sản phẩm họ mua đứt.

- **Ví dụ về Intent :**
    - "Máy Kindle của tôi bị treo màn hình, làm sao để reset được?"
    - "Loa Alexa không kết nối được wifi ."
    - "Tôi muốn gia hạn gói đăng ký xem phim của hệ thống."

### 2. Sự khác biệt về Taxonomy (Cấu trúc phân loại)

Hệ thống phân loại của hai nhóm này không đồng nhất và có những đặc điểm riêng biệt:

- **Đặc điểm của 3P (Bên thứ ba):**
    - Ý định của khách hàng thường xoay quanh các quy trình mua sắm truyền thống.
    - Cấu trúc được chia theo các cấp độ từ khái quát đến chi tiết (Level 1, Level 2, Level 3).
    - *Ví dụ:* Một luồng phân loại điển hình có thể là: Vấn đề đơn hàng (Level 1) → Trả hàng/Hoàn tiền (Level 2) → Kiểm tra trạng thái (Level 3).
- **Đặc điểm của 1P (Chính chủ):**
    - Yêu cầu các dịch vụ tùy chỉnh và chuyên sâu hơn do tính chất phức tạp của thiết bị và sản phẩm số.
    - Tập trung vào việc xử lý sự cố (troubleshooting) và hỗ trợ kỹ thuật.
    - Cấu trúc phân loại phản ánh các danh mục sản phẩm cụ thể như: Thiết bị (Device), Kỹ thuật số (Digital), Máy tính bảng thông minh (Smart tablet), hoặc Cửa hàng ứng dụng (App store).

**Tại sao bài báo chia ra như vậy?**
Vì khách hàng hỏi về "Lỗi kết nối Bluetooth" (1P) cần một quy trình xử lý hoàn toàn khác với việc "Chưa nhận được hàng" (3P). Việc chia **Vertical** giúp mô hình REIC truy xuất đúng "kho kiến thức" (Knowledge Base) của từng mảng, tránh việc đưa ra hướng dẫn sửa máy Kindle cho một người đang muốn trả lại cái áo thun.