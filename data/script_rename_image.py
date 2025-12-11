import os
import unicodedata
import re

# ============================
# CONFIG
# ============================
FOLDER = r"\\192.168.0.171\Public\hcns\drive server\BMC LONG AN\5. IT\data-training-ai\8. Kinh doanh\1. Sản phẩm, cây trồng, nấm bệnh.I\3. Bộ sản phẩm\image-sp"
NEW_PREFIX = "sheet-sp"   # prefix mới muốn thay
TARGET_PREFIX = "sheet-sp "  # prefix cũ cần tìm

# ============================
# HÀM CHUẨN HOÁ TÊN FILE
# ============================

def normalize_prefix(text: str) -> str:
    """
    Chuẩn hoá chuỗi có dấu thành không dấu + thay khoảng trắng bằng '-'.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    no_diacritic = "".join([c for c in nfkd if not unicodedata.combining(c)])
    no_diacritic = no_diacritic.lower().strip()
    no_diacritic = re.sub(r"[^a-z0-9]+", "-", no_diacritic)
    no_diacritic = re.sub(r"-+", "-", no_diacritic).strip("-")
    return no_diacritic


def main():
    # CHUẨN HOÁ prefix cũ & prefix mới
    old_prefix_norm = normalize_prefix(TARGET_PREFIX)
    new_prefix_norm = normalize_prefix(NEW_PREFIX)

    print(f"OLD PREFIX (normalized): {old_prefix_norm}")
    print(f"NEW PREFIX (normalized): {new_prefix_norm}")

    for filename in os.listdir(FOLDER):
        old_path = os.path.join(FOLDER, filename)

        if not os.path.isfile(old_path):
            continue

        # Chỉ xử lý file ảnh phổ biến
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            continue

        # Normalize tên file để so sánh prefix
        filename_norm = normalize_prefix(filename)

        # Kiểm tra xem tên file có prefix cũ trước '_page_' không
        match = re.match(rf"({old_prefix_norm})(.*)", filename_norm)
        if not match:
            continue

        # Tìm đúng vị trí "_Page_" trong tên gốc để cắt
        split_index = filename.lower().find("_page_")
        if split_index == -1:
            continue  # không phải file dạng mong muốn

        original_suffix = filename[split_index:]  # giữ nguyên phần Page_164_Image...

        # tạo tên file mới
        new_filename = f"{new_prefix_norm}{original_suffix}"
        new_path = os.path.join(FOLDER, new_filename)

        print(f"RENAMING:\n  {filename}\n  → {new_filename}\n")

        os.rename(old_path, new_path)

    print("Hoàn tất đổi tên!")


if __name__ == "__main__":
    main()
