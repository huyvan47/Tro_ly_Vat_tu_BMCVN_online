import pandas as pd
import ast
import json
import unicodedata
from pathlib import Path

# =====================
# CONFIG
# =====================
INPUT_CSV  = "data-kd-1-4-tags-v2-entity-type_fixed_tags_v2.csv"
OUTPUT_CSV = "data-kd-1-4-tags-v2-entity-type_fixed_tags_v2_no_accent.csv"
COL_NAME   = "tags_v2"

ENC = "utf-8-sig"
# =====================


def remove_vietnamese_tone(text: str) -> str:
    """Bỏ dấu tiếng Việt"""
    if not text:
        return text
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_tag(tag: str) -> str:
    """
    Giữ prefix trước ':'
    Bỏ dấu phần sau ':'
    """
    tag = tag.strip()
    if ":" not in tag:
        return remove_vietnamese_tone(tag)

    prefix, value = tag.split(":", 1)
    value = remove_vietnamese_tone(value)
    return f"{prefix}:{value}"


def normalize_tags_cell(cell: str) -> str:
    """
    Xử lý 1 ô tags_v2 (list dạng string)
    """
    cell = (cell or "").strip()
    if not cell:
        return cell

    try:
        tags = ast.literal_eval(cell)
        if not isinstance(tags, list):
            return cell
    except Exception:
        return cell

    new_tags = []
    for t in tags:
        if not isinstance(t, str):
            new_tags.append(t)
        else:
            new_tags.append(normalize_tag(t))

    # Ghi lại dạng JSON list chuẩn
    return json.dumps(new_tags, ensure_ascii=False)


# =====================
# MAIN
# =====================
def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, encoding=ENC)

    if COL_NAME not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{COL_NAME}'")

    df[COL_NAME] = df[COL_NAME].apply(normalize_tags_cell)

    out_path = Path(OUTPUT_CSV)
    df.to_csv(out_path, index=False, encoding=ENC)

    print(f"✅ Done. Normalized '{COL_NAME}' (no accent) for entire KB.")
    print(f"📄 Output CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
