def format_direct_doc_answer(user_query: str, primary_doc: dict) -> str:
    """
    Trả lời trực tiếp bằng nội dung tài liệu (extractive) + gợi ý.
    """
    q = (primary_doc.get("question", "") or "").strip()
    a = (primary_doc.get("answer", "") or "").strip()

    out = []
    out.append("Nội dung phù hợp nhất trong tài liệu:")

    if q:
        out.append(f"- Mục liên quan: {q}")

    if a:
        out.append("")
        out.append(a)

    return "\n".join(out).strip()