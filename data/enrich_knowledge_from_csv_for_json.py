import json
from pathlib import Path
from openai import OpenAI

# ==============================
# CẤU HÌNH
# ==============================

MODEL_NAME = "gpt-4.1-mini"
client = OpenAI(api_key="...")

JSON_PATH = Path("sections_nam_benh_vn.json")
OUT_CSV_PATH = Path("sections_nam_benh_vn_chunks.csv")

# ==============================
# PROMPT LLM – TỰ SUY LUẬN
# ==============================

system_prompt = """
Bạn là chuyên gia nông nghiệp và data engineer. 

NHIỆM VỤ:
- Tự suy luận:
  - Đối tượng cây trồng / bệnh
  - Category phù hợp
  - ID_PREFIX phù hợp theo nội dung
- Sau đó chuyển đoạn văn bản thành các dòng CSV theo schema:

id,question,answer,category,tags,alt_questions

YÊU CẦU:
- id:
  - Tự tạo ID_PREFIX theo nội dung (ví dụ: ic_top, citrus_scab, rose_disease…)
  - Sau đó đánh số tăng dần: _chunk_01, _chunk_02, _qa_01…
- question: câu hỏi tự nhiên, rõ ràng, tiếng Việt.
- answer: giải thích chi tiết, giữ CƠ CHẾ – BƯỚC – LƯU Ý – AN TOÀN nếu có.
- category: tự chọn category phù hợp theo nội dung.
- tags: 3–6 từ khóa, phân tách bằng dấu |
- alt_questions_v2: 2–5 biến thể câu hỏi, phân tách bằng dấu |

PHÂN LOẠI:
- 2–4 dòng CHUNK
- 2–4 dòng Q&A “vì sao – khi nào – lưu ý – cơ chế”

OUTPUT:
- CHỈ xuất ra CSV
- CÓ header
- KHÔNG giải thích ngoài CSV
"""

# ==============================
# HÀM PHỤ
# ==============================

def extract_text_from_item(item: dict):
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None

def build_prompt(raw: str) -> str:
    return f"""
Đây là nội dung gốc:

{raw}

Hãy sinh dữ liệu CSV đúng yêu cầu.
"""

def call_model_for_text(raw: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(raw)},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ==============================
# MAIN
# ==============================

def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {JSON_PATH}")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    all_csv_lines = []
    header_written = False

    for idx, item in enumerate(data, start=1):
        text = extract_text_from_item(item)
        if not text:
            print(f"[BỎ QUA] Section #{idx} không có text")
            continue

        print(f"Đang xử lý section #{idx}...")

        csv_text = call_model_for_text(text)
        lines = [l.strip() for l in csv_text.splitlines() if l.strip()]

        if not lines:
            continue

        if not header_written:
            all_csv_lines.extend(lines)
            header_written = True
        else:
            if lines[0].lower().startswith("id,question"):
                lines = lines[1:]
            all_csv_lines.extend(lines)

    if not all_csv_lines:
        print("⚠ Không có dữ liệu CSV nào được sinh.")
        return

    final_csv = "\n".join(all_csv_lines)
    OUT_CSV_PATH.write_text(final_csv, encoding="utf-8")

    print(f"\n✅ HOÀN TẤT! Đã lưu tại: {OUT_CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
