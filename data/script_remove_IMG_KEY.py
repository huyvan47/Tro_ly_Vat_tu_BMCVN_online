import re
import pandas as pd
from pathlib import Path

# ==========================
# CONFIG
# ==========================
INPUT_CSV = "data-vat-tu-full-done-enriched-25-11.csv"          # file CSV gốc
OUTPUT_CSV = "output_with_img.csv"   # file CSV sau khi xử lý
ANSWER_COL = "answer"            # tên cột chứa nội dung
IMG_KEYS_COL = "img_keys"        # tên cột mới

# Regex tìm (IMG_KEY: xxx)
IMG_KEY_PATTERN = re.compile(r"\(IMG_KEY:\s*([^)]+)\)")


def extract_img_keys_and_clean_answer(answer: str):
    """
    Từ một chuỗi answer:
    - Lấy ra list img_keys trong các cặp (IMG_KEY: xxx)
    - Xoá các đoạn (IMG_KEY: xxx) khỏi answer, dọn khoảng trắng
    """
    if not isinstance(answer, str):
        return "", answer

    # Tìm tất cả IMG_KEY
    keys = IMG_KEY_PATTERN.findall(answer)

    # Xoá các đoạn (IMG_KEY: xxx)
    cleaned = IMG_KEY_PATTERN.sub("", answer)

    # Dọn khoảng trắng thừa
    cleaned = re.sub(r"\s{2,}", " ", cleaned)   # gộp nhiều space thành 1
    cleaned = cleaned.replace(" ,", ",").strip()

    # Gộp img_keys thành 1 chuỗi, ngăn bởi |
    keys_str = "|".join(k.strip() for k in keys if k.strip())

    return keys_str, cleaned


def main():
    # Đọc CSV
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # Nếu chưa có cột img_keys thì tạo
    if IMG_KEYS_COL not in df.columns:
        df[IMG_KEYS_COL] = ""

    # Xử lý từng dòng
    img_keys_list = []
    cleaned_answers = []

    for idx, val in df[ANSWER_COL].items():
        keys_str, cleaned = extract_img_keys_and_clean_answer(val)
        img_keys_list.append(keys_str)
        cleaned_answers.append(cleaned)

    # Gán lại vào DataFrame
    df[IMG_KEYS_COL] = img_keys_list
    df[ANSWER_COL] = cleaned_answers

    # Ghi ra file mới
    out_path = Path(OUTPUT_CSV)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã xử lý xong. File mới: {out_path.resolve()}")


if __name__ == "__main__":
    main()
