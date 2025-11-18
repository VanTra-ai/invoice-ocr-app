import streamlit as st
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io
import pytesseract
import math

# --- 0. CẤU HÌNH ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    st.error("⚠️ Kiểm tra lại đường dẫn cài đặt Tesseract OCR")

st.set_page_config(page_title="📦 AI Nhập Kho Hóa Đơn", layout="wide", page_icon="🧾")

LABEL_COLORS = {
    "ItemName": (139, 0, 0), "ItemNameValue": (255, 69, 0),
    "Quantity": (0, 100, 0), "QuantityValue": (50, 205, 50),
    "UnitPrice": (0, 0, 139), "UnitPriceValue": (30, 144, 255),
    "Amount": (75, 0, 130), "AmountValue": (255, 0, 255),
    "Other": (192, 192, 192)
}


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "./final_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
        processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=True)
        return model, processor, device
    except Exception as e:
        st.error(f"❌ Lỗi tải model: {e}")
        return None, None, None


model, processor, device = load_model()


# --- HÀM TẠO BẢNG SẢN PHẨM ---
def create_invoice_table(tokens, labels, bboxes, tokenizer, image_width, image_height):
    HEADER_MAP = {
        "ItemNameValue": "Tên sản phẩm",
        "QuantityValue": "Số lượng",
        "UnitPriceValue": "Đơn giá",
        "AmountValue": "Thành tiền"
    }

    # --- 1. GOM TOKEN THÀNH ENTITY (B- VÀ I- TƯƠNG ỨNG) ---
    merged_entities = []
    current_ent = None

    for token, label, box in zip(tokens, labels, bboxes):
        if token in tokenizer.all_special_tokens:
            continue

        clean_token = token.replace("Ġ", " ").strip()  # LayoutLMv3 dùng Ġ cho khoảng trắng

        if label.startswith("B-"):
            # Lưu entity cũ
            if current_ent:
                merged_entities.append(current_ent)

            clean_label = label.replace("B-", "")

            if clean_label not in HEADER_MAP:
                current_ent = None
                continue

            current_ent = {
                "text": clean_token,
                "label": clean_label,
                "box": box  # bounding box gốc
            }

        elif label.startswith("I-"):
            clean_label = label.replace("I-", "")

            if current_ent and current_ent["label"] == clean_label:
                # Ghép token vào
                current_ent["text"] += " " + clean_token

                # Mở rộng bounding box để bao phủ toàn entity
                x1, y1, x2, y2 = current_ent["box"]
                b1, b2, b3, b4 = box
                current_ent["box"] = [min(x1, b1), min(y1, b2), max(x2, b3), max(y2, b4)]

    # Lưu cuối cùng
    if current_ent:
        merged_entities.append(current_ent)

    # Sau khi ghép, nếu không có entity nào → trả bảng rỗng
    if not merged_entities:
        return pd.DataFrame(columns=["Tên sản phẩm", "Số lượng", "Đơn giá", "Thành tiền"])

    # Làm sạch text
    for ent in merged_entities:
        ent["text"] = " ".join(ent["text"].split())  # bỏ khoảng trắng thừa

    # --- 2. CHUYỂN BOUNDING BOX VỀ PIXEL ---
    for ent in merged_entities:
        x1, y1, x2, y2 = ent["box"]
        ent["box"] = [
            x1 * image_width / 1000,
            y1 * image_height / 1000,
            x2 * image_width / 1000,
            y2 * image_height / 1000
        ]
        ent["center_y"] = (ent["box"][1] + ent["box"][3]) / 2
        ent["col_name"] = HEADER_MAP[ent["label"]]

    # --- 3. SẮP XẾP THEO TỌA ĐỘ Y ---
    merged_entities.sort(key=lambda x: x["center_y"])

    # --- 4. GHÉP THÀNH CÁC DÒNG SẢN PHẨM ---
    final_rows = []
    current = {"Tên sản phẩm": "", "Số lượng": "", "Đơn giá": "", "Thành tiền": "", "y": 0}

    THRESH = 22  # ngưỡng lệch Y để tách dòng

    for ent in merged_entities:
        col = ent["col_name"]
        y = ent["center_y"]
        text = ent["text"]

        # Nếu lệch dòng → tạo dòng mới
        if current["Tên sản phẩm"] and abs(y - current["y"]) > THRESH:
            final_rows.append({k: current[k] for k in ["Tên sản phẩm", "Số lượng", "Đơn giá", "Thành tiền"]})
            current = {"Tên sản phẩm": "", "Số lượng": "", "Đơn giá": "", "Thành tiền": "", "y": y}

        # Ghép text theo cột
        current[col] += (" " + text) if current[col] else text
        current["y"] = (current["y"] + y) / 2 if current["y"] else y

    # Append dòng cuối
    if current["Tên sản phẩm"]:
        final_rows.append({k: current[k] for k in ["Tên sản phẩm", "Số lượng", "Đơn giá", "Thành tiền"]})

    df = pd.DataFrame(final_rows)

    # Ép kiểu string để tránh Streamlit khóa cell
    for col in ["Số lượng", "Đơn giá", "Thành tiền"]:
        df[col] = df[col].astype("string")

    return df

# --- VẼ HÌNH ---
def draw_image(image, bboxes, labels, width, height):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for box, label in zip(bboxes, labels):
        clean_label = label.replace("B-", "").replace("I-", "")
        if clean_label not in LABEL_COLORS:
            continue

        color = LABEL_COLORS[clean_label]
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width / 1000, x2 * width / 1000
        y1, y2 = y1 * height / 1000, y2 * height / 1000

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if label.startswith("B-"):
            draw.rectangle([x1, y1 - 20, x1 + 70, y1], fill=color)
            draw.text((x1 + 2, y1 - 20), clean_label, fill="white", font=font)

    return image


# --- GIAO DIỆN ---
st.title("🧾 Demo AI Trích Xuất Hóa Đơn (LayoutLMv3)")

if "df_products" not in st.session_state:
    st.session_state.df_products = pd.DataFrame(columns=["Tên sản phẩm", "Số lượng", "Đơn giá", "Thành tiền"])
if "annotated_img" not in st.session_state:
    st.session_state.annotated_img = None

uploaded_file = st.file_uploader("📤 Tải ảnh hóa đơn", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Ảnh gốc")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

# --- PHÂN TÍCH ---
if st.button("🚀 Phân tích hóa đơn"):
    if uploaded_file:
        with st.spinner("Đang xử lý…"):

            encoding = processor(image, return_tensors="pt", truncation=True, max_length=512)
            for k, v in encoding.items():
                encoding[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**encoding)

            preds = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding.bbox.squeeze().tolist()
            input_ids = encoding.input_ids.squeeze().tolist()
            id2label = model.config.id2label

            labels = [id2label[p] for p in preds]
            tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

            df_new = create_invoice_table(tokens, labels, token_boxes, processor.tokenizer, image.width, image.height)

            st.session_state.df_products = df_new
            st.session_state.annotated_img = draw_image(image.copy(), token_boxes, labels, image.width, image.height)

with col1:
    if st.session_state.annotated_img is not None:
        st.subheader("🎯 Nhận diện")
        st.image(st.session_state.annotated_img, use_container_width=True)

with col2:
    st.subheader("📋 Phiếu nhập kho (sửa được)")

    df = st.session_state.df_products

    if not df.empty:

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Tên sản phẩm": st.column_config.TextColumn(width="large"),
            }
        )

        st.session_state.df_products = edited_df

        if st.button("💾 Lưu kho"):
            final = edited_df.copy()

            # Convert về số
            for col in ["Số lượng", "Đơn giá", "Thành tiền"]:
                final[col] = pd.to_numeric(final[col], errors="coerce")

            st.success("Dữ liệu đã sẵn sàng:")
            st.json(final.to_dict(orient="records"))
    else:
        st.warning("Chưa có dữ liệu sản phẩm.")
