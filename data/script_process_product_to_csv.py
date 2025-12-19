import json
import re
import pandas as pd
from pathlib import Path

from openai import OpenAI

# ========= CONFIG =========
EXCEL_PATH = "input-product-catologue-p6.xlsx"
SHEET_NAME = 0               # hoặc "Sheet1"
TEXT_COL = "product_text"    # đổi theo file bạn
OUT_CSV = "input-product-catologue-p6.csv"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

ID_PREFIX_DEFAULT = "sp-tbvtv"   # bạn có thể đổi

SYSTEM_PROMPT = r"""
Bạn là công cụ chuyển đổi mô tả sản phẩm thuốc BVTV thành DỮ LIỆU RAG DẠNG MASTER ENTRY.

MỤC TIÊU:
Mỗi sản phẩm chỉ tạo DUY NHẤT 1 bản ghi (1 row CSV).
Khi người dùng tìm bất kỳ tên/nhãn nào, phải truy xuất được TOÀN BỘ thông tin.

QUY TẮC BẮT BUỘC:
1) Chỉ dùng thông tin có trong INPUT. Không bịa, không suy đoán.
2) Trả về DUY NHẤT 1 JSON hợp lệ theo schema bên dưới.
3) KHÔNG chia chunk, KHÔNG tạo alias riêng.
4) Gộp toàn bộ thông tin vào trường "answer" theo cấu trúc:
   - Nhóm sản phẩm
   - Các tên / nhãn tương đương
   - Hoạt chất
   - Nhóm thuốc
   - Hãng sản xuất
   - Đặc tính – công dụng
   - Đối tượng
   - Phạm vi
   - Liều dùng – cách pha – lượng nước
   - Thời điểm – hiệu lực
   - Lưu ý an toàn
   - Trường hợp cỏ già (nếu có)
   - Kinh nghiệm phối trộn (nếu có)
5) Trường "question" phải chứa TẤT CẢ tên/nhãn chính.
6) tags phải chứa:
   - group:<slug nhóm sản phẩm>
   - alias:<slug> cho từng nhãn
   - ai:<hoạt chất>
   - ai_rate:<hàm lượng>
   - type:<herbicide/fungicide/insecticide/other>
7) alt_questions: 5–8 câu, có chứa tên nhãn để tăng recall.
8) img_keys luôn là chuỗi rỗng "".

SCHEMA JSON PHẢI TRẢ:
{
  "row": {
    "id": "string",
    "question": "string",
    "answer": "string",
    "category": "string",
    "tags": "string",
    "alt_questions": ["string", "..."],
    "img_keys": ""
  }
}
""".strip()

def safe_json_extract(text: str) -> dict:
    """
    Cố gắng parse JSON. Nếu model lỡ bọc thêm text, sẽ trích khối JSON đầu tiên.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # fallback: tìm khối {...} lớn nhất
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))

def build_user_prompt(product_text: str, id_prefix: str, row_index: int) -> str:
    return (
        "INPUT_TEXT:\n"
        f"<<<{product_text}>>>\n\n"
        f"ID_PREFIX: {id_prefix}\n"
        f"ROW_INDEX: {row_index}\n"
        "Hãy tạo JSON theo đúng schema."
    )

def process_one(client: OpenAI, product_text: str, row_index: int, id_prefix: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INPUT_TEXT:\n<<<{product_text}>>>\n\nROW_INDEX: {row_index}"},
        ],
    )

    content = resp.choices[0].message.content
    obj = safe_json_extract(content)

    row = obj.get("row")
    if not isinstance(row, dict):
        raise ValueError("Model returned invalid 'row' object.")

    return {
        "id": row.get("id", f"{id_prefix}_{row_index:04d}_master"),
        "question": normalize_for_csv(row.get("question", "")),
        "answer": normalize_for_csv(row.get("answer", "")),
        "category": normalize_for_csv(row.get("category", "")),
        "tags": row.get("tags", "").strip(),
        "alt_questions": json.dumps(row.get("alt_questions", []), ensure_ascii=False),
        "img_keys": ""
    }


def normalize_for_csv(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)  # gộp nhiều dòng trống
    text = text.replace("\n", ". ")       # newline -> ". "
    text = re.sub(r"\s{2,}", " ", text)   # gộp khoảng trắng
    return text.strip()

def main():
    client = OpenAI(api_key="...")

    # File không có header, mỗi sản phẩm nằm ở 1 hàng cột A
    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=SHEET_NAME,
        header=None,      # không có tiêu đề
        usecols="A",      # chỉ lấy cột A
        dtype=str
    )
    df.columns = ["product_text"]

    # làm sạch dòng trống
    df["product_text"] = df["product_text"].fillna("").astype(str).str.strip()
    df = df[df["product_text"] != ""].reset_index(drop=True)

    all_rows = []
    for i, val in enumerate(df[TEXT_COL].fillna("").astype(str).tolist(), start=1):
        product_text = val.strip()
        if not product_text:
            continue

        row = process_one(client, product_text, i, ID_PREFIX_DEFAULT)
        all_rows.append(row)

    out_df = pd.DataFrame(all_rows, columns=["id","question","answer","category","tags","alt_questions","img_keys"])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done. Wrote {len(out_df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
