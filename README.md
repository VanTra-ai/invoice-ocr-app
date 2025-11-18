# 🧾 AI-OCR INVOICE ENTRY SYSTEM (LayoutLMv3)

Hệ thống giúp tự động trích xuất thông tin sản phẩm từ **hóa đơn giấy / ảnh chụp** và tự động đổ vào **Phiếu nhập kho/kế toán**.  
Ứng dụng sử dụng **Tesseract OCR + LayoutLMv3** để hiểu bố cục (layout) và nhận dạng bảng sản phẩm trên hóa đơn.

---

# 1. 📌 TỔNG QUAN DỰ ÁN

### 🎯 Mục tiêu chính
- Tự động hóa quy trình nhập kho.
- Giảm tối đa thao tác nhập liệu thủ công.
- Hạn chế sai sót do con người.
- Cho phép chỉnh sửa dữ liệu trực tiếp trước khi xác nhận.

### 🧠 Công nghệ sử dụng
- LayoutLMv3 (HuggingFace Transformers)
- PyTorch
- Streamlit (UI Demo)
- Tesseract OCR
- Custom Post-processing (Line Clustering + Token Reconstruction)

### 📚 Dữ liệu huấn luyện
- JSON chuẩn DocVQA / FUNSD dạng Việt hóa  
- Các file: `VN_0003.json`, `VN_0004.json`, …

---

# 2. 🏗️ KIẾN TRÚC & LUỒNG XỬ LÝ

## 2.1. Kiến trúc tổng thể

```bash
  [Client UI - Streamlit]
  ↓ Upload Image
  [OCR Engine - Tesseract]
  ↓ Tokens + Bounding Boxes
  [AI Engine - LayoutLMv3]
  ↓ Token Classification
  [Post-processing]
  ↓ Line Reconstruction
  [Editable Invoice Table]
  ↓ Export JSON
```

---

# 3. ⚙️ CÀI ĐẶT MÔI TRƯỜNG (LOCAL DEMO)

## 3.1. 🔥 Cài đặt Tesseract OCR (BẮT BUỘC)

Ứng dụng **không thể hoạt động** nếu thiếu Tesseract vì LayoutLMv3 cần:
- Text OCR  
- Bounding boxes  
- Thứ tự token  

---

## 🟦 Bước 1 — Tải bản Tesseract đúng

Truy cập:

👉 https://github.com/UB-Mannheim/tesseract/wiki

Tải file: `tesseract-ocr-w64-setup-v5.x.x.xxxx.exe`  Đây là bản 64-bit mới nhất và hỗ trợ tốt tiếng Việt.

---

## 🟩 Bước 2 — Chọn gói tiếng Việt khi cài đặt

Trong bước **Select Components** của trình cài đặt:

**Quan trọng: bạn phải mở rộng 2 mục sau (nhấn dấu +)**:
- Additional script data (download)
- Additional language data (download)

Sau đó **tìm và tích chọn**: `Vietnamese (vie)`


⚠ Nếu bạn không bật gói Vietnamese → Tesseract sẽ **không đọc được dấu tiếng Việt**.

---

## 🟩 Bước 3 — Hoàn tất cài đặt

Nhấn Install → Đợi hoàn tất.

---

## 🟩 Bước 4 — Thêm Tesseract vào PATH

Thư mục mặc định: `C:\Program Files\Tesseract-OCR\`


### Cách thêm vào PATH:
1. Mở Start → “Environment Variables”
2. Chọn **Edit the system environment variables**
3. Nhấn **Environment Variables**
4. Trong **System variables**, mở **Path**
5. Nhấn **New**
6. Dán: `C:\Program Files\Tesseract-OCR\`


---

## 🟩 Bước 5 — Kiểm tra Tesseract đã hoạt động

Mở CMD và chạy:

```bash
tesseract --version
```
Nếu hiện version → OK.

## 🔧 Bước 6 — Cấu hình Tesseract trong `app.py`

Sau khi cài đặt Tesseract thành công, bạn cần chỉ định đường dẫn đầy đủ đến file `tesseract.exe` trong dự án Python.

Thông thường Tesseract được cài vào: `C:\Program Files\Tesseract-OCR\tesseract.exe`
Nhưng nếu bạn cài vào ổ khác (ví dụ ổ D: hoặc thư mục tùy chỉnh), bạn phải sửa lại:

Ví dụ:
```python
pytesseract.pytesseract.tesseract_cmd = r'D:\Tools\Tesseract-OCR\tesseract.exe'
```

---

## 📁 3.2. Cấu trúc Project & Dependencies

Dự án nên có cấu trúc như sau:

```bash
invoice-ocr-app/
├── app.py
├── README.md
├── requirements.txt
└── final_model/
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```

---

### 📦 Cài đặt Dependencies

Chạy lệnh sau trong terminal:

```sh
pip install -r requirements.txt
```

---

# 4. ▶️ Chạy Ứng Dụng Demo

Trong terminal, chạy:

```sh
streamlit run app.py
```

Ứng dụng sẽ mở tại: 👉 `http://localhost:8501`

---

# 5. 📤 Xuất dữ liệu (JSON)

Sau khi người dùng chỉnh sửa bảng sản phẩm và nhấn “💾 Lưu kho”, dữ liệu sẽ được chuyển thành JSON chuẩn, ví dụ:

```json
[
  {
    "Tên sản phẩm": "Cam",
    "Số lượng": 12,
    "Đơn giá": 17500,
    "Thành tiền": 210000
  }
]
```
