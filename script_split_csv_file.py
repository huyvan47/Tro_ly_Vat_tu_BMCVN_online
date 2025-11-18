import csv
import json

INPUT_CSV = "muc-luc-vat-tu-phu_paraphrased.csv"         # file CSV nguồn
OUTPUT_JSONL = "output.jsonl"   # file jsonl xuất ra

# Tên cột trong file CSV
QUESTION_COL = "question"
ANSWER_COL = "answer"

SYSTEM_PROMPT = (
    "Bạn là Trợ lý Vật tư BMCVN. Giọng chuyên nghiệp, trả lời ngắn gọn, "
    "có bullet khi cần."
)

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f_in, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    
    reader = csv.DictReader(f_in)

    for row in reader:
        question = (row.get(QUESTION_COL) or "").strip()
        answer   = (row.get(ANSWER_COL) or "").strip()

        # Bỏ qua dòng trống
        if not question or not answer:
            continue

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }

        # Ghi 1 dòng JSON vào file .jsonl
        f_out.write(json.dumps(data, ensure_ascii=False))
        f_out.write("\n")

print("✔️ Đã convert CSV sang JSONL xong!")
