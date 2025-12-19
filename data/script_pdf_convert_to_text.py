# ================== CONFIG ==================
import pdfplumber
import pandas as pd

# ========================
# CONFIG
# ========================
PDF_PATH = r"Catologue-BMC-1-2.pdf"
OUTPUT_XLSX = r"Catologue-BMC-1-2.xlsx"

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
