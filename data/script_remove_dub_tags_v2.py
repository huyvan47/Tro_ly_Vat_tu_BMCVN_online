import pandas as pd
import ast

# ====== CONFIG ======
INPUT_CSV = "data-kd-1-4-tags-v2-entity-type.csv"
COL_NAME = "tags_v2"          # đổi thành tên cột của anh
OUTPUT_TXT = "unique_tags_v2.txt"
# ====================

df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")

unique = set()

for cell in df[COL_NAME]:
    cell = (cell or "").strip()
    if not cell:
        continue

    # cell dạng '["a","b"]' -> parse thành list
    try:
        items = ast.literal_eval(cell)
        if not isinstance(items, list):
            continue
    except Exception:
        # nếu cell không phải format list chuẩn thì bỏ qua hoặc xử lý riêng
        continue

    for t in items:
        t = str(t).strip()
        if t:
            unique.add(t)   # lọc trùng theo toàn chuỗi (action:apply ≠ action:spraying)

# Xuất ra file, sort cho dễ nhìn
unique_sorted = sorted(unique)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for t in unique_sorted:
        f.write(t + "\n")

print(f"✅ Unique tags: {len(unique_sorted)}")
print(f"📄 Saved to: {OUTPUT_TXT}")
