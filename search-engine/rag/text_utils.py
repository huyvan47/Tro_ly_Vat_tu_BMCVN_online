import re
def extract_img_keys(text: str):
    return re.findall(r'\(IMG_KEY:\s*([^)]+)\)', text or "")

def is_listing_query(q: str) -> bool:
    t = (q or "").lower()
    return any(x in t for x in [
        "các loại", "những loại", "những", "bao nhiêu loại", "tất cả", "liệt kê",
        "kể tên", "tổng", "có bao nhiêu", "gồm",
        "các bệnh", "những bệnh", "bệnh nào", "gồm những bệnh nào"
    ])