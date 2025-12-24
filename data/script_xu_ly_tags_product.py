import pandas as pd
import ast
import json

# =====================
# CONFIG
# =====================
INPUT_CSV  = "data-kd-1-4-tags-v2-entity-type.csv"
OUTPUT_CSV = "data-kd-1-4-tags-v2-entity-type_fixed_tags_v2.csv"
COL = "tags_v2"

# =====================
# LOGIC
# =====================
def fix_tags_v2_only_second(cell: str) -> str:
    cell = (cell or "").strip()
    if not cell:
        return cell

    try:
        tags = ast.literal_eval(cell)
        if not isinstance(tags, list) or len(tags) < 2:
            return cell
    except Exception:
        return cell

    # Chỉ xử lý nếu phần tử đầu là entity:product
    if tags[0] != "entity:product":
        return cell

    new_tags = tags.copy()

    # 👉 CHỈ xử lý phần tử thứ 2 (index = 1)
    second = new_tags[1]
    if isinstance(second, str) and ":" in second:
        _, suffix = second.split(":", 1)
        new_tags[1] = f"product:{suffix}"

    # Các phần tử từ index >=2 giữ nguyên
    return json.dumps(new_tags, ensure_ascii=False)


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{COL}'")

    df[COL] = df[COL].apply(fix_tags_v2_only_second)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Fixed tags_v2 (only 2nd prefix) → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
