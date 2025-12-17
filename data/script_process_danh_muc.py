import pandas as pd
import numpy as np
import re

xlsx_path = r"Danh mục 2025.xlsx"   # đổi path nếu cần
sheet_name = "DM2026"               # sheet của file bạn

out_csv = r"dmtbvtv_rag.csv"

# 1) Đọc file thô để tìm đúng dòng header (do Excel có phần tiêu đề/merge cell)
raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str)

keywords = [
    "Rank", "TT", "NGÀNH", "NHÓM",
    "HOẠT CHẤT", "TÊN THƯƠNG PHẨM",
    "TỔ CHỨC ĐỀ NGHỊ", "APPLICANT",
    "GROUP", "NUM"
]

best_score, header_row = -1, 0
for i in range(min(300, len(raw))):
    row_text = " | ".join(raw.iloc[i].fillna("").astype(str).tolist())
    score = sum(1 for k in keywords if k.lower() in row_text.lower())
    if score > best_score:
        best_score, header_row = score, i

# 2) Đọc lại với header đúng
df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row, dtype=str)

# 3) Clean tên cột cho ổn định
def clean_col(c):
    c = str(c).strip().replace("\n", " ")
    c = re.sub(r"\s+", " ", c)
    return c

df.columns = [clean_col(c) for c in df.columns]

# 4) Hàm lấy giá trị an toàn
def s(v):
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()

# 5) Tạo CSV theo schema kho tri thức của bạn
rows = []
for idx, r in df.iterrows():
    rid = f"dmtbvtv_row_{idx+1:04d}"

    trade = s(r.get("TÊN THƯƠNG PHẨM (TRADE NAME)", ""))
    question = f"Tên thương phẩm: {trade}" if trade else "Tên thương phẩm:"

    applicant = s(r.get("TỔ CHỨC ĐỀ NGHỊ ĐĂNG KÝ (APPLICANT)", ""))
    common = s(r.get("HOẠT CHẤT/THUỐC BVTV KỸ THUẬT (COMMON NAME)", ""))

    # alt_questions = applicant + common (có thể trống)
    alt_questions = " | ".join([x for x in [applicant, common] if x])

    answer_lines = [
        f"Ngành: {s(r.get('NGÀNH',''))}",
        f"Nhóm: {s(r.get('NHÓM',''))}",
        f"Hoạt chất: {common}",
        f"Tên thương phẩm: {trade}",
        f"Đối tượng/cây trồng: {s(r.get('ĐỐI TƯỢNG PHÒNG TRỪ (PEST/ CROP)',''))}",
        f"Đơn vị đăng ký: {applicant}",
        f"Rank: {s(r.get('Rank',''))}",
        f"TT: {s(r.get('TT',''))}",
        f"GROUP: {s(r.get('GROUP',''))}",
        f"NUM: {s(r.get('NUM',''))}",
    ]
    answer = "\n".join(answer_lines).strip()

    rows.append({
        "id": rid,
        "question": question,
        "answer": answer,
        "category": "dmtbvtv",
        "tags": "",
        "alt_questions": alt_questions,
        "img_keys": ""
    })

out_df = pd.DataFrame(rows, columns=[
    "id","question","answer","category","tags","alt_questions","img_keys"
])

out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Done. Rows: {len(out_df)}. Output: {out_csv}. Header row detected: {header_row}")
