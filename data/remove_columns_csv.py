import csv

# ================== CONFIG ==================

INPUT_CSV = "data-vat-tu-full-done-enriched-05-12-f1-100-done.csv"              # File CSV gốc
OUTPUT_CSV = "data-vat-tu-full-done-enriched-05-12-f1-100-done-remove-colum.csv"   # File CSV sau khi xóa cột

# Danh sách các cột cần XÓA (đúng tên header)
COLUMNS_TO_REMOVE = [
    "answer_enriched",
    "alt_questions",
    # thêm bao nhiêu cột cũng được
]

# ============================================


def remove_columns_csv(input_path, output_path, remove_cols):
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)

        # Lọc ra các cột cần giữ lại
        remaining_fields = [
            field for field in reader.fieldnames
            if field not in remove_cols
        ]

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=remaining_fields)
            writer.writeheader()

            for row in reader:
                new_row = {
                    key: value for key, value in row.items()
                    if key in remaining_fields
                }
                writer.writerow(new_row)

    print("✅ Hoàn tất!")
    print("➡ File đầu vào:", input_path)
    print("➡ File sau khi xóa cột:", output_path)
    print("🗑 Cột đã bị xóa:", remove_cols)


if __name__ == "__main__":
    remove_columns_csv(INPUT_CSV, OUTPUT_CSV, COLUMNS_TO_REMOVE)
