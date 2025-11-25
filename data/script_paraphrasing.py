import pandas as pd
from openai import OpenAI
import json

client = OpenAI(api_key="...")

# Load file RAG
df = pd.read_csv("data-vat-tu-full-done-enriched-25-11.csv", encoding="utf-8")

def generate_paraphrase(question):
    prompt = f"""
Hãy tạo 4 câu hỏi biến tấu khác nhau dựa trên câu hỏi gốc:
"{question}"

YÊU CẦU:
- Không đổi ý nghĩa.
- Viết theo nhiều cách hỏi của người dùng.
- Giữ nguyên các mã chai, ví dụ: cha240-06, 1000-03, thung-q10,...
- Trả kết quả dưới dạng list JSON.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    text = resp.choices[0].message.content.strip()

    try:
        arr = json.loads(text)
    except:
        arr = [text]

    # Nối thành 1 chuỗi
    return " | ".join(arr)

# Tạo paraphrase cho từng câu
df["alt_questions"] = df["question"].apply(generate_paraphrase)

# Lưu file mới
df.to_csv("data-vat-tu-full-done-enriched-25-11-PARA.csv", index=False, encoding="utf-8")

print("Done! File đã tạo:")
print("data-vat-tu-full-done-enriched-25-11-PARA.csv")
