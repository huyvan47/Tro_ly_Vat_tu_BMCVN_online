def detect_answer_mode(user_query: str, primary_doc: dict, is_listing: bool) -> str:
    if is_listing:
        return "listing"

    text = (user_query + " " + primary_doc.get("question", "")).lower()
    cat = primary_doc.get("category", "").lower()

    if any(kw in text for kw in [
        "quy trình", "quy trinh", "các bước"
    ]) or "quy_trinh" in cat:
        return "verbatim"

    if any(kw in text for kw in [
        "bệnh", "triệu chứng", "dấu hiệu","nứt thân", "xì mủ", "thối rễ", "cháy lá", "thán thư", "ghẻ"
    ]) or "benh" in cat:
        return "disease"

    if any(kw in text for kw in [
        "thuốc", "đặc trị", "sc", "wp", "ec", "sl", "dùng để làm gì", "tác dụng", "hoạt chất", "chữa","trị","trừ"
    ]) or "san_pham" in cat or "thuoc" in cat:
        return "product"

    if any(kw in text for kw in [
        "cách làm", "hướng dẫn", "chính sách", "thương phẩm", "tên thương phẩm"
        "làm thế nào để", "phương pháp", "thí nghiệm",
        "phân lập", "giám định", "lây bệnh trong phòng thí nghiệm"
    ]) or "quy_trinh" in cat:
        return "procedure"
    
    return "general"