import pandas as pd
import re

# ====== CONFIG ======
INPUT_CSV  = "dmtbvtv_rag.csv"
OUTPUT_CSV = "dmtbvtv_rag_clean.csv"

# Cột cần xử lý
REMOVE_COLON_IN = ["question"]
REMOVE_NEWLINE_IN = ["answer"]

# ====================

df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

def clean_text(val, remove_colon=False, remove_newline=False):
    if not isinstance(val, str):
        return val

    if remove_colon:
        val = val.replace(":", "")

    if remove_newline:
        # thay mọi loại xuống dòng bằng 1 dấu cách
        val = re.sub(r"[\r\n]+", " ", val)
        val = re.sub(r"\s+", " ", val).strip()

    return val

for col in df.columns:
    if col in REMOVE_COLON_IN:
        df[col] = df[col].apply(lambda x: clean_text(x, remove_colon=True))

    if col in REMOVE_NEWLINE_IN:
        df[col] = df[col].apply(lambda x: clean_text(x, remove_newline=True))

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Done. Clean file saved to: {OUTPUT_CSV}")
