import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "gpt-4.1-mini"
INPUT_XLSX = "tai_lieu_cay_lua.xlsx"        # file có 2 cột: page, text
OUTPUT_CSV = "tai_lieu_cay_lua_rag.csv"

client = OpenAI(api_key="...")

# ==============================
# PROMPT RAG TEMPLATE
# ==============================

SYSTEM_PROMPT_TEMPLATE = """
Bạn là chuyên gia xây dựng dữ liệu RAG và data engineer.

NHIỆM VỤ:
- Nhận MỘT ĐOẠN NỘI DUNG TEXT gốc từ file PDF (đã OCR hoặc extract).
- Tự phân tích ý nghĩa chính của đoạn.
- Tạo 1–3 CHUNK RAG và 2–4 Q&A xoay quanh nội dung đoạn.

SCHEMA CSV:
id,question,answer,category,tags,alt_questions

YÊU CẦU:
- id:
  - Bắt đầu bằng tiền tố: {id_prefix}
  - Sau đó thêm _chunk_01, _qa_01,...
- question:
  - Câu hỏi tự nhiên, rõ ràng, tiếng Việt, liên quan đoạn text.
- answer:
  - Diễn giải lại đầy đủ, đúng ý và có hệ thống.
- category:
  - Một từ/cụm từ ngắn không dấu (ví dụ: "kien_thuc_pdf", "chuong_sach")
- tags:
  - 3–7 từ khóa không dấu, phân cách bằng |
- alt_questions:
  - 2–5 biến thể câu hỏi khác, phân cách bằng |

ĐỊNH DẠNG OUTPUT:
- Trả về CSV TEXT duy nhất, có header:
  id,question,answer,category,tags,alt_questions
- Không dùng Markdown và không bọc trong ```
"""


# ==============================
#  HÀM HỖ TRỢ
# ==============================

def slugify_prefix(text: str) -> str:
    """
    Tạo id_prefix từ 10–20 ký tự đầu của text.
    """
    if not text:
        return "chunk"

    # lấy 20 ký tự đầu
    short = text.strip()[:20].lower()
    short = re.sub(r"[^a-z0-9]+", "_", short)
    short = re.sub(r"_+", "_", short).strip("_")

    if not short:
        return "chunk"

    return short


def call_llm(text_block: str, id_prefix: str) -> str:
    """
    Gọi LLM sinh chunk CSV từ 1 text-block.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(id_prefix=id_prefix)

    user_prompt = (
        "Dưới đây là nội dung một đoạn text từ PDF:\n\n"
        f"{text_block}\n\n"
        "Hãy sinh dữ liệu kiến thức RAG theo đúng yêu cầu và đúng schema."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return resp.choices[0].message.content.strip()


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_excel(INPUT_XLSX)

    if "text" not in df.columns:
        raise ValueError("File Excel phải có cột 'text'.")

    all_rows = []
    header_written = False

    for idx, row in df.iterrows():

        text_block = str(row.get("text", "")).strip()
        if not text_block:
            continue

        print(f"Đang xử lý dòng {idx+1}...")

        id_prefix = slugify_prefix(text_block)
        csv_text = call_llm(text_block, id_prefix)

        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        if not lines:
            continue

        # Lấy header lần đầu
        if not header_written:
            all_rows.append(lines[0])
            header_written = True

        # Các dòng tiếp theo bỏ header nếu có
        body_lines = (
            lines[1:] if lines[0].lower().startswith("id,question") else lines
        )
        all_rows.extend(body_lines)

    out_path = Path(OUTPUT_CSV)
    out_path.write_text("\n".join(all_rows), encoding="utf-8-sig")

    print("\nHoàn tất!")
    print(f"Đã ghi file CSV RAG vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
