def format_direct_doc_answer(user_query: str, primary_doc: dict, suggestions_text: str) -> str:
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

    if suggestions_text and suggestions_text.strip() and "(không có)" not in suggestions_text:
        out.append("")
        out.append("Câu hỏi liên quan:")
        out.append(suggestions_text)

    return "\n".join(out).strip()