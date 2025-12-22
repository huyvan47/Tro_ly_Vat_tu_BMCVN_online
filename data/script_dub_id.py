import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "kb-audit/check-backbone/data-kd-full-1_with_entity_type-2-nam-benh-viet-nam.csv"
OUTPUT_CSV = ROOT / "output-data-kd-full-1_with_entity_type-2-nam-benh-viet-nam-update-category.csv"

df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# Thêm hậu tố toàn cục dựa trên index
df["id"] = df["id"] + "__" + df.index.astype(str).str.zfill(6)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("✅ Fixed duplicate ids by appending global index.")
