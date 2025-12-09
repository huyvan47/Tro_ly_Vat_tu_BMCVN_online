import pdfplumber
import re
import json
from pathlib import Path

# ================== CONFIG ==================
PDF_PATH = r"\\192.168.0.171\Public\hcns\drive server\BMC LONG AN\5. IT\data-training-ai\8. Kinh doanh\1. Sản phẩm, cây trồng, nấm bệnh.I\1. Tài liệu cây trồng\1. Quy trình sầu riêng\Sầu riêng chuyên sâu\3. Sâu bệnh sầu riêng.pdf"
OUTPUT_JSON = "sections_sau_benh_sau_rieng.json"

# Regex nhận dạng heading dạng:
# 1 Giới thiệu
# 1.1 Tài liệu tham khảo
# 10.4.1 Sinh sản hữu tính ...
HEADING_PATTERN = re.compile(
    r"^(\d{1,2}(?:\.\d+)*)(?:\s+)(.+)$"  # group 1: số mục, group 2: tiêu đề
)

# Nếu cần bỏ qua một số dòng gây nhiễu (ví dụ "Nội dung", "Lời nói đầu" không đánh số)
IGNORE_PATTERNS = [
    re.compile(r"^Nội dung$", re.IGNORECASE),
    re.compile(r"^Lời nói đầu$", re.IGNORECASE),
]


def is_ignored(line: str) -> bool:
    """Kiểm tra xem một dòng có nằm trong danh sách bỏ qua không."""
    for pat in IGNORE_PATTERNS:
        if pat.match(line):
            return True
    return False


def heading_level(heading_id: str) -> int:
    """
    Tính 'level' của heading dựa trên số lượng phần sau khi split bằng dấu chấm.
    Ví dụ: '1' -> 1, '1.2' -> 2, '10.4.1' -> 3
    """
    return len(heading_id.split("."))


def extract_sections_from_pdf(pdf_path: str):
    sections = []
    current_section = None
    section_counter = 0

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = text.splitlines()

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                if is_ignored(line):
                    continue

                m = HEADING_PATTERN.match(line)
                if m:
                    # Gặp heading mới
                    heading_id = m.group(1).strip()
                    title = m.group(2).strip()

                    # Đóng section cũ lại
                    if current_section is not None:
                        current_section["end_page"] = page_idx
                        # Join nội dung thành 1 block text
                        current_section["text"] = "\n".join(current_section["text_lines"]).strip()
                        del current_section["text_lines"]
                        sections.append(current_section)

                    # Tạo section mới
                    section_counter += 1
                    current_section = {
                        "index": section_counter,
                        "heading_id": heading_id,        # ví dụ: "10.4.1"
                        "title": title,                  # ví dụ: "Sinh sản hữu tính ..."
                        "level": heading_level(heading_id),
                        "start_page": page_idx,
                        "end_page": page_idx,           # tạm, sẽ cập nhật khi gặp heading sau
                        "text_lines": [],               # tạm lưu từng dòng, lát join lại
                    }
                else:
                    # Dòng nội dung thuộc section hiện tại
                    if current_section is not None:
                        current_section["text_lines"].append(line)
                    else:
                        # Nội dung trước khi gặp heading đầu tiên (nếu muốn lưu lại)
                        # Có thể bỏ qua hoặc gom vào một section đặc biệt
                        pass

        # Kết thúc file, đóng section cuối cùng nếu có
        if current_section is not None:
            current_section["end_page"] = len(pdf.pages)
            current_section["text"] = "\n".join(current_section["text_lines"]).strip()
            del current_section["text_lines"]
            sections.append(current_section)

    return sections


def main():
    sections = extract_sections_from_pdf(PDF_PATH)

    # Ghi ra JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"Đã tách được {len(sections)} section, lưu vào: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
