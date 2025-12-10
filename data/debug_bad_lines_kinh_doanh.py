import csv
from pathlib import Path

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data-kinh-doanh_FIXED-3.csv"
OUTPUT = ROOT / "data-kinh-doanh/data-kinh-doanh_FIXED-3.csv"

EXPECTED_COLS = 7

# ==============================
#        PROCESS CSV
# ==============================

def fix_csv():
    print(f"Đang đọc file: {INPUT}")

    with open(INPUT, "r", encoding="utf-8", newline="") as fin, \
         open(OUTPUT, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        print(f"Header có {len(header)} cột:", header)
        print("Bắt đầu xử lý...\n")

        line_no = 1
        fixed_count = 0
        bad_lines = 0

        for row in reader:
            line_no += 1
            col_count = len(row)

            # Nếu số cột đúng → ghi lại luôn
            if col_count == EXPECTED_COLS:
                writer.writerow(row)
                continue

            # Nếu ít hơn → thêm cột trống
            if col_count < EXPECTED_COLS:
                missing = EXPECTED_COLS - col_count
                row.extend([""] * missing)
                fixed_count += 1
                writer.writerow(row)
                continue

            # Nếu nhiều hơn 7 cột → báo lỗi (phải sửa tay)
            if col_count > EXPECTED_COLS:
                bad_lines += 1
                print("===== LỖI: DÒNG NHIỀU HƠN 7 CỘT =====")
                print(f"Dòng: {line_no}")
                print(f"Số cột thực tế: {col_count}")
                print(f"Nội dung row: {row}")
                print()
                # vẫn ghi để không mất dữ liệu
                writer.writerow(row)

        print("\n==============================")
        print(f"✔ Tổng dòng đã sửa (thêm cột trống): {fixed_count}")
        print(f"⚠ Dòng có nhiều hơn 7 cột (cần kiểm tra tay): {bad_lines}")
        print(f"👉 File CSV mới đã lưu tại: {OUTPUT}")
        print("==============================")

if __name__ == "__main__":
    fix_csv()

