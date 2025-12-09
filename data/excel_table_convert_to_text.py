import re
import csv
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ==============================
# CẤU HÌNH
# ==============================
MODEL_NAME = "gpt-4.1-mini"   # hoặc model khác anh muốn
INPUT_XLSX = "sp-delta.xlsx"
OUTPUT_CSV = "sp-delta_rag.csv"

# Đặt API key trong biến môi trường: OPENAI_API_KEY
client = OpenAI(api_key="...")

# ==============================
# MAP TÊN CỘT TRONG EXCEL
# ==============================
"""
Anh mở file Excel, xem tên cột ở hàng đầu rồi sửa lại cho khớp.

Ví dụ nếu file có cột:
- "Ten SP", "Hoat chat", "Quy cach BMC", "Quy cach Phuc Thinh",
  "Cong dung", "Cay trong", "Luu y"

thì anh sửa lại như dưới cho đúng chính tả.
"""

COLUMN_MAP = {
    "name": "Ten SP",
    "ai_active": "Hoat chat",
    "pack": "Quy cach ban",
    "uses": "Cong dung",
    "highlight": "Dac diem noi bat",
    "competitor": "San pham doi thu",
    "note": "Ghi chu",
}

# ==============================
# PROMPT CHO LLM
# ==============================

SYSTEM_PROMPT_TEMPLATE = """
Bạn là chuyên gia thuốc bảo vệ thực vật và data engineer.

NHIỆM VỤ:
- Nhận mô tả chi tiết về MỘT sản phẩm thuốc BVTV.
- Tự suy luận:
  - Đối tượng cây trồng / cỏ / sâu bệnh chính
  - Category phù hợp (ví dụ: "thuoc_co", "thuoc_tru_benh", "thuoc_sau", "phan_bon_la"...)
  - Tags ngắn gọn.
- Sau đó sinh dữ liệu kiến thức dạng CSV để dùng cho RAG.

SCHEMA CSV:
id,question,answer,category,tags,alt_questions_v2

YÊU CẦU:
- id:
  - Luôn bắt đầu bằng tiền tố: {id_prefix}
  - Sau đó đánh số tăng dần: _chunk_01, _chunk_02, _qa_01, _qa_02...
- question:
  - Câu hỏi tự nhiên, rõ ràng, tiếng Việt.
- answer:
  - Trả lời chi tiết, mạch lạc, tiếng Việt.
  - BẮT BUỘC giữ đầy đủ thông tin quan trọng trong mô tả:
    + Hoạt chất, nồng độ
    + Quy cách bán
    + Cây trồng / đối tượng cỏ / sâu / bệnh
    + Cách dùng, liều lượng nếu có
    + Các LƯU Ý quan trọng (cây không được phun, giai đoạn không được dùng...)
- category:
  - Một từ / cụm từ ngắn gọn, không dấu: ví dụ "thuoc_co", "thuoc_tru_benh", "thuoc_sau".
- tags:
  - 3–7 từ khóa, không dấu, phân tách bằng dấu |
  - Ví dụ: "maruka_5ec|thuoc_co|co_la_hep|dau_tuong|lac"
- alt_questions_v2:
  - 2–5 biến thể câu hỏi khác nhau, tiếng Việt, phân tách bằng dấu |
  - Nội dung xoay quanh: sản phẩm dùng làm gì, cho cây nào, liều lượng, lưu ý an toàn...

PHÂN LOẠI BẢN GHI:
- Tạo 1–2 dòng dạng CHUNK mô tả tổng quan sản phẩm:
  - Ví dụ: mô tả đầy đủ thành phần, công dụng, cây trồng, lưu ý an toàn.
- Tạo 2–4 dòng dạng Q&A GIẢI THÍCH / HƯỚNG DẪN:
  - Ví dụ: Maruka 5EC dùng cho cây gì? | Có phun được trên bắp không? | Những lưu ý an toàn khi dùng? ...

ĐỊNH DẠNG OUTPUT:
- CHỈ xuất ra CSV TEXT, có header:
  id,question,answer,category,tags,alt_questions_v2
- Không dùng Markdown, không bọc ```.
- Escape dấu phẩy và xuống dòng đúng chuẩn CSV (bọc chuỗi trong "..." nếu cần).
"""

def build_product_description(row):
    def get(col_key):
        col_name = COLUMN_MAP[col_key]
        val = row.get(col_name, "")
        if pd.isna(val):
            return ""
        return str(val).strip()

    name = get("name")
    ai_active = get("ai_active")
    pack = get("pack")
    uses = get("uses")
    highlight = get("highlight")
    competitor = get("competitor")
    note = get("note")

    parts = []

    if name:
        parts.append(f"Sản phẩm {name} của công ty Tỷ Phúc Thịnh.")

    if ai_active:
        parts.append(f"Hoạt chất: {ai_active}.")

    if pack:
        parts.append(f"Quy cách bán: {pack}.")

    if uses:
        parts.append(f"Công dụng: {uses}.")

    if highlight:
        parts.append(f"Đặc điểm nổi bật: {highlight}.")

    if competitor:
        parts.append(f"Sản phẩm đối thủ: {competitor}.")

    if note:
        parts.append(f"Lưu ý: {note}.")

    return "\n".join(parts).strip()

def slugify(text):
    """Tạo id_prefix đẹp từ tên sản phẩm."""
    text = text.lower()
    # thay ký tự đặc biệt / khoảng trắng bằng _
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "sp_bv_tv"
    return text


def call_llm_for_product(description, id_prefix):
    """Gọi LLM trả về CSV text cho MỘT sản phẩm."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(id_prefix=id_prefix)
    user_prompt = (
        "Dưới đây là mô tả chi tiết của MỘT sản phẩm thuốc BVTV.\n\n"
        f"{description}\n\n"
        "Hãy sinh dữ liệu CSV theo đúng yêu cầu."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    csv_text = resp.choices[0].message.content.strip()
    return csv_text


def main():
    df = pd.read_excel(INPUT_XLSX)
    all_rows = []
    header_written = False

    for idx, row in df.iterrows():
        
        name_col = COLUMN_MAP["name"]
        name_val = row.get(name_col, "")
        if pd.isna(name_val) or not str(name_val).strip():
            # bỏ các dòng trống tên sản phẩm
            continue
        
        name_str = str(name_val).strip()
        id_prefix = slugify(name_str)

        description = build_product_description(row)
        
        if not description:
            continue

        print(f"Đang xử lý sản phẩm: {name_str} (id_prefix={id_prefix})")

        csv_text = call_llm_for_product(description, id_prefix)

        # Tách các dòng CSV, xử lý header
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        if not lines:
            continue

        if not header_written:
            all_rows.append(lines[0])  # header
            header_written = True

        # các dòng sau header
        data_lines = lines[1:] if lines[0].lower().startswith("id,question") else lines
        all_rows.extend(data_lines)

    # Ghi ra file CSV cuối cùng
    out_path = Path(OUTPUT_CSV)
    out_path.write_text("\n".join(all_rows), encoding="utf-8-sig")
    print(f"\nĐã ghi file RAG CSV vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
