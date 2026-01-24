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

    blocks = []

    # =========================
    # MODE 1: FORMULA → enrich theo CHEMICAL (giữ logic cũ)
    # =========================
    if answer_mode == "formula":
        # 1️⃣ ƯU TIÊN: chemical từ tags
        chemicals = extract_chemicals_from_matched_tags(
            any_tags=any_tags,
            must_tags=must_tags,
        )

        # 2️⃣ Fallback: từ answer text
        if not chemicals:
            c = extract_primary_chemical_from_answer(answer_text)
            if c:
                chemicals = [c]

        if not chemicals:
            return answer_text

        for chem in chemicals[:1]:  # formula → 1 chemical là đủ
            try:
                info = query_global_knowledge(
                    client=client,
                    chemical=chem,
                    user_query=user_query,
                    model="gpt-4.1",
                    max_tokens=900,
                )
                if info:
                    blocks.append(
                        f"### Kiến thức nền về hoạt chất {chem.upper()} (ngoài tài liệu nội bộ)\n"
                        f"{info.strip()}\n"
                    )
            except Exception:
                continue

    # =========================
    # MODE 2: PRODUCT → enrich theo CÂU HỎI (không gắn nhãn chemical)
    # =========================
    elif answer_mode == "product":
        try:
            info = query_global_knowledge(
                client=client,
                chemical=user_query,   # không ép chemical
                user_query=user_query,
                model="gpt-4.1",
                max_tokens=900,
            )
            if info:
                blocks.append(info.strip())
        except Exception:
            pass

    # =========================
    # Không có gì để enrich
    # =========================
    if not blocks:
        return answer_text

    # =========================
    # Appendix chung
    # =========================
    appendix = (
        "\n\n---\n\n"
        "## BỔ SUNG KIẾN THỨC NỀN (tham khảo khoa học, ngoài tài liệu nội bộ)\n"
        "Lưu ý: Phần dưới đây là kiến thức nền mang tính tham khảo, "
        "không thay thế hướng dẫn trên nhãn/khuyến cáo nhà sản xuất.\n\n"
        + "\n".join(blocks)
    )

    return answer_text.rstrip() + appendix
