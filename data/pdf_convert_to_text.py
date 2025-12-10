# ================== CONFIG ==================
import pdfplumber
import pandas as pd

# ========================
# CONFIG
# ========================
PDF_PATH = r"\\192.168.0.171\Public\hcns\drive server\BMC LONG AN\5. IT\data-training-ai\8. Kinh doanh\1. Sản phẩm, cây trồng, nấm bệnh.I\1. Tài liệu cây trồng\ĐỘI LONG AN- CÂY LÚA .pdf"
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
