import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ==============================
# CẤU HÌNH
# ==============================
MODEL_NAME = "gpt-4.1-mini"   # hoặc model khác anh muốn
INPUT_XLSX = "sau-hai-page-5.xlsx"
OUTPUT_CSV = "sau-hai-page-5_rag.csv"

# Đặt API key trực tiếp hoặc dùng biến môi trường OPENAI_API_KEY
client = OpenAI(api_key="...")

# ==============================
# MAP TÊN CỘT TRONG EXCEL
# ==============================
"""
Anh mở file sp-delta.xlsx, kiểm tra đúng y nguyên header.
Nếu khác (ví dụ 'Sau hai' không dấu, hay 'Cong thuc phoi hop'),
sửa lại các string bên phải cho khớp.
"""

COLUMN_MAP = {
    "pest": "Sâu hại",
    "group": "Nhóm",                  # nếu cột tên khác, sửa ở đây
    "density": "Mật độ",
    "formula": "Công thức phối hợp",
    "note": "Lưu ý",
}


# ==============================
# PROMPT CHO LLM
# ==============================

SYSTEM_PROMPT_TEMPLATE = """
Bạn là chuyên gia thuốc bảo vệ thực vật và data engineer.

NHIỆM VỤ:
- Nhận THÔNG TIN MỘT DÒNG từ bảng khuyến cáo phòng trừ sâu hại
  (gồm: sâu hại, nhóm sâu, mật độ, công thức phối hợp, lưu ý).
- Tự suy luận:
  - Đối tượng sâu hại chính
  - Nhóm cây trồng/điều kiện áp dụng nếu có thể suy ra
  - Category phù hợp (ví dụ: "phac_do_sau_hai", "phac_do_ray_nau"...)
  - Tags ngắn gọn.
- Sau đó sinh dữ liệu kiến thức dạng CSV để dùng cho RAG.

SCHEMA CSV:s
id,question,answer,category,tags,alt_questions

YÊU CẦU:
- id:
  - Luôn bắt đầu bằng tiền tố: {id_prefix}
  - Sau đó đánh số tăng dần: _chunk_01, _chunk_02, _qa_01, _qa_02...
- question:
  - Câu hỏi tự nhiên, rõ ràng, tiếng Việt.
- answer:
  - Trả lời chi tiết, mạch lạc, tiếng Việt.
  - BẮT BUỘC giữ đầy đủ thông tin quan trọng trong mô tả:
    + Tên sâu hại / đối tượng
    + Nhóm sâu
    + Mật độ xuất hiện
    + Công thức phối hợp thuốc: tên thuốc, lượng, nước, hiệu quả
    + Các LƯU Ý quan trọng (có thể thay thuốc nào, điều kiện áp dụng...)
- category:
  - Một từ / cụm từ ngắn gọn, không dấu: ví dụ "phac_do_sau_hai", "phac_do_ray_nau".
- tags:
  - 3–7 từ khóa, không dấu, phân tách bằng dấu |
  - Ví dụ: "ray_nau|ray_lung_trang|pham_do|cong_thuc_phoi_hop"
- alt_questions:
  - 2–5 biến thể câu hỏi khác nhau, tiếng Việt, phân tách bằng dấu |
  - Nội dung xoay quanh: sâu hại nào, mật độ nào dùng công thức này, tỷ lệ phối trộn,
    lưu ý khi sử dụng, có thể thay thuốc gì...

PHÂN LOẠI BẢN GHI:
- Tạo 1–2 dòng dạng CHUNK mô tả tổng quan phác đồ cho sâu hại đó:
  - Ví dụ: mô tả đầy đủ sâu hại, nhóm, mật độ, công thức, lưu ý.
- Tạo 2–4 dòng dạng Q&A GIẢI THÍCH / HƯỚNG DẪN:
  - Ví dụ:
    + Khi rầy nâu mật độ cao thì dùng công thức gì?
    + Công thức Exami + Binhfos phối trộn ra sao?
    + Có được thay Binhfos bằng thuốc nào khác không?
    + Khi nào ưu tiên dùng lưu dẫn mạnh?

ĐỊNH DẠNG OUTPUT:
- CHỈ xuất ra CSV TEXT, có header:
  id,question,answer,category,tags,alt_questions
- Không dùng Markdown, không bọc ```.
- Escape dấu phẩy và xuống dòng đúng chuẩn CSV (bọc chuỗi trong "..." nếu cần).
"""


# ==============================
# HÀM HỖ TRỢ
# ==============================

def build_record_description(row: pd.Series) -> str:
    """Gộp các cột thành 1 đoạn mô tả chuẩn cho MỘT phác đồ sâu hại."""
    def get(col_key: str) -> str:
        col_name = COLUMN_MAP[col_key]
        # row là Series, dùng .get để tránh lỗi nếu thiếu cột
        val = row.get(col_name, "")
        if pd.isna(val):
            return ""
        return str(val).strip()

    pest = get("pest")
    group = get("group")
    density = get("density")
    formula = get("formula")
    note = get("note")

    parts = []

    if pest:
        parts.append(f"Đối tượng sâu hại: {pest}.")
    if group:
        parts.append(f"Nhóm sâu hại: {group}.")
    if density:
        parts.append(f"Mật độ xuất hiện: {density}.")
    if formula:
        parts.append(f"Công thức phối hợp khuyến cáo: {formula}.")
    if note:
        parts.append(f"Lưu ý khi áp dụng: {note}.")

    return "\n".join(parts).strip()


def slugify(text: str) -> str:
    """Tạo id_prefix đẹp từ tên sâu hại."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "phac_do_sau_hai"
    return text


def call_llm_for_record(description: str, id_prefix: str) -> str:
    """Gọi LLM trả về CSV text cho MỘT dòng phác đồ."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(id_prefix=id_prefix)

    user_prompt = (
        "Dưới đây là thông tin chi tiết của MỘT dòng phác đồ phối hợp thuốc "
        "để phòng trừ sâu hại:\n\n"
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

    return resp.choices[0].message.content.strip()


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_excel(INPUT_XLSX)

    # Anh có thể bật dòng này để kiểm tra đúng header:
    # print(df.columns)

    all_rows = []
    header_written = False

    for idx, row in df.iterrows():
        # Lấy cột sâu hại làm "tên chính" để tạo id_prefix
        pest_col = COLUMN_MAP["pest"]
        pest_val = row.get(pest_col, "")
        if pd.isna(pest_val) or not str(pest_val).strip():
            # bỏ các dòng trống sâu hại
            continue

        pest_str = str(pest_val).strip()
        id_prefix = slugify(pest_str)

        description = build_record_description(row)
        if not description:
            continue

        print(f"Đang xử lý sâu hại: {pest_str} (id_prefix={id_prefix})")

        csv_text = call_llm_for_record(description, id_prefix)

        # Tách các dòng CSV, xử lý header
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        if not lines:
            continue

        # Header
        if not header_written:
            all_rows.append(lines[0])
            header_written = True

        # Các dòng sau header
        data_lines = lines[1:] if lines[0].lower().startswith("id,question") else lines
        all_rows.extend(data_lines)

    # Ghi ra file CSV cuối cùng
    out_path = Path(OUTPUT_CSV)
    out_path.write_text("\n".join(all_rows), encoding="utf-8-sig")
    print(f"\nĐã ghi file RAG CSV vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
