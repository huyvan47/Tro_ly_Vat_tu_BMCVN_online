import pandas as pd
import json
import re

# 1. Đọc file PARA gốc
df = pd.read_csv("data-vat-tu-full-done-enriched-25-11-PARA.csv", encoding="utf-8")

def extract_clean_alt(raw: str) -> str:
    """
    Nhận vào alt_questions dạng 'văn + ```json [...] ```'
    → Trả về: 'q1 | q2 | q3 | q4'
    """
    if not isinstance(raw, str):
        return ""

    # Bỏ hoàn toàn phần code block ```...``` cho chắc
    # (nhưng vẫn giữ bản raw để tìm JSON bên trong)
    # Thật ra ta chỉ cần phần [ ... ] bên trong thôi
    # nên chỉ cần tìm đoạn [ ... ] trong toàn string
    match = re.search(r"\[(.*?)\]", raw, flags=re.DOTALL)
    if not match:
        return ""

    array_content = "[" + match.group(1) + "]"

    # Thử parse JSON
    try:
        arr = json.loads(array_content)
        # arr là list các string
        qs = [q.strip() for q in arr if isinstance(q, str)]
        return " | ".join(qs)
    except Exception:
        # Fallback: parse từng dòng có dấu "
        lines = match.group(1).splitlines()
        qs = []
        for line in lines:
            m = re.search(r'"(.*?)"', line)
            if m:
                qs.append(m.group(1).strip())
        return " | ".join(qs)

# 2. Tạo cột alt_questions_clean từ alt_questions
df["alt_questions_clean"] = df["alt_questions"].apply(extract_clean_alt)

# 3. Lưu ra file mới để dùng embed
df.to_csv("data-vat-tu-full-done-enriched-25-11-PARA-CLEAN.csv",
          index=False,
          encoding="utf-8")

print("DONE → data-vat-tu-full-done-enriched-25-11-PARA-CLEAN.csv")
