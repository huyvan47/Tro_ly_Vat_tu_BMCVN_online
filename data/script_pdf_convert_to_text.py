# ================== CONFIG ==================
import pdfplumber
import pandas as pd

# ========================
# CONFIG
# ========================
PDF_PATH = r"\\192.168.0.171\Public\hcns\drive server\BMC LONG AN\5. IT\data-training-ai\8. Kinh doanh\2. 8 bài kỹ năng Kinh doanh\8. Lập Kế hoạch đi Thị trường, thanh quyết toán.pdf"
OUTPUT_XLSX = r"output.xlsx"

# ========================
# PROCESS PDF
# ========================
rows = []

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        rows.append({
            "page": i,
            "text": text
        })

# ========================
# EXPORT TO EXCEL
# ========================
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)

print(f"Đã xuất toàn bộ nội dung PDF sang Excel: {OUTPUT_XLSX}")
