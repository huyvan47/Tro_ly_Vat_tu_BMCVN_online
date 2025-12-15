import re
def extract_img_keys(text: str):
    return re.findall(r'\(IMG_KEY:\s*([^)]+)\)', text or "")

def is_listing_query(q: str) -> bool:
    t = (q or "").lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê",
        "kể tên", "bao nhiêu", "tổng", "có bao nhiêu", "gồm",
        "các bệnh", "những bệnh", "bệnh nào", "gồm những bệnh nào"
    ])

def extract_codes_from_query(text: str):
    # ví dụ: cha240-06, 450-02, cha240-asmil-01...
    return re.findall(r'\b[\w]*\d[\w-]*-\d[\w-]*\b', text or "")