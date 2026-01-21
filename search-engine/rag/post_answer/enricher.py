from rag.post_answer.chemical_extractor import (
    extract_chemicals_from_matched_tags,
    extract_primary_chemical_from_answer,
)
from rag.post_answer.global_knowledge import query_global_knowledge
from rag.post_answer.detectors import should_enrich_post_answer


def enrich_answer_if_needed(
    *,
    client,
    user_query: str,
    answer_text: str,
    answer_mode: str,
    any_tags: list,
    must_tags: list,
    route: str = "RAG",
) -> str:

    if not should_enrich_post_answer(
        user_query=user_query,
        answer_text=answer_text,
        answer_mode=answer_mode,
        route=route,
    ):
        return answer_text

    # 1️⃣ ƯU TIÊN TUYỆT ĐỐI: chemical từ tags_filter
    chemicals = extract_chemicals_from_matched_tags(
        any_tags=any_tags,
        must_tags=must_tags,
    )

    # 2️⃣ Fallback nhẹ: từ answer text (rất hiếm khi cần)
    if not chemicals:
        c = extract_primary_chemical_from_answer(answer_text)
        if c:
            chemicals = [c]

    if not chemicals:
        # Không xác định được chemical → KHÔNG enrich
        return answer_text

    blocks = []

    for chem in chemicals[:1]:  # product Q&A → 1 chemical là đủ
        try:
            info = query_global_knowledge(
                client=client,
                chemical=chem,
                user_query=user_query,
                model="gpt-4.1-mini",
                max_tokens=900,
            )
            if info:
                blocks.append(
                    f"### Kiến thức nền về hoạt chất {chem.upper()} (ngoài tài liệu nội bộ)\n{info.strip()}\n"
                )
        except Exception:
            continue

    if not blocks:
        return answer_text

    appendix = (
        "\n\n---\n\n"
        "## BỔ SUNG KIẾN THỨC NỀN (tham khảo khoa học, ngoài tài liệu nội bộ)\n"
        "Lưu ý: Phần dưới đây là kiến thức nền theo hoạt chất, không thay thế hướng dẫn trên nhãn/khuyến cáo nhà sản xuất.\n\n"
        + "\n".join(blocks)
    )

    return answer_text.rstrip() + appendix
