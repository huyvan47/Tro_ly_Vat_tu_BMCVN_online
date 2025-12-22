import pandas as pd
from pathlib import Path

# =====================
# CONFIG
# =====================
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "kb-audit/check-backbone/data-kd-full-1_with_entity_type-2-nam-benh-viet-nam.csv"
OUTPUT_CSV = ROOT / "output-data-kd-full-1_with_entity_type-2-nam-benh-viet-nam-update-category.csv"

# 2 tham số chỉ định đoạn dòng (0-based, inclusive)
START_ROW = 1
END_ROW   = 101

NEW_CATEGORY = "disease_profile"

# Thêm tag vào trước cột tags (nếu rỗng thì không thêm)
ADD_TAG = ""   # <-- để "" nếu không muốn thêm

# Tuỳ chọn: loại trùng tag
DEDUP = True
# =====================


def prepend_tag(existing: str, tag_to_add: str, dedup: bool = True) -> str:
    """
    Chèn tag_to_add vào đầu chuỗi tags hiện có, phân tách bằng '|'.
    - Nếu tag_to_add rỗng: trả về existing nguyên trạng
    - Tránh tạo '||'
    - Nếu dedup=True: không thêm nếu đã có tag
    """
    existing = (existing or "").strip()
    tag_to_add = (tag_to_add or "").strip()

    # Không thêm nếu biến rỗng
    if not tag_to_add:
        return existing

    # Chuẩn hoá tags hiện có -> list
    existing_tags = [t.strip() for t in existing.split("|") if t.strip()] if existing else []

    if dedup:
        # So sánh theo lower để tránh trùng do hoa/thường
        lower_set = {t.lower() for t in existing_tags}
        if tag_to_add.lower() in lower_set:
            return "|".join(existing_tags)

    # Prepend
    new_tags = [tag_to_add] + existing_tags
    return "|".join(new_tags)


# Đọc CSV
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# Validate index
max_index = len(df) - 1
if START_ROW < 0 or END_ROW > max_index or START_ROW > END_ROW:
    raise ValueError(
        f"Khoảng dòng không hợp lệ: START_ROW={START_ROW}, END_ROW={END_ROW}, max={max_index}"
    )

# 1) Update category theo đoạn
df.loc[START_ROW:END_ROW, "category"] = NEW_CATEGORY

# 2) Update tags theo đoạn (chỉ khi cần)
if "tags" not in df.columns:
    df["tags"] = ""

df.loc[START_ROW:END_ROW, "tags"] = df.loc[START_ROW:END_ROW, "tags"].apply(
    lambda x: prepend_tag(x, ADD_TAG, dedup=DEDUP)
)

# Ghi file
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(
    f"✅ Done. Updated rows {START_ROW} → {END_ROW}. "
    f"Category='{NEW_CATEGORY}', ADD_TAG='{ADD_TAG}'."
)
print(f"📄 Output: {OUTPUT_CSV}")
