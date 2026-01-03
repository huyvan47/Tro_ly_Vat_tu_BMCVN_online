import re

def route_query(client, user_query: str) -> str:
    """
    Trả về:
    - "RAG"    : ưu tiên dùng tài liệu nội bộ
    - "GLOBAL" : để model tự trả lời bằng kiến thức nền

    Chiến lược:
    1) Heuristic cứng (ổn định, rẻ, giảm sai)
    2) Nếu chưa quyết được -> hỏi LLM router
    3) Parse output theo equality, fallback an toàn
    """

    q = (user_query or "").strip().lower()

    # ---------------------------
    # 1) Heuristic: intent định nghĩa/khái niệm -> GLOBAL
    # ---------------------------
    definition_signals = [
        r"\b(là gì|nghĩa là gì|định nghĩa|khái niệm)\b",
        r"\b(bao gồm|gồm những|gồm các|thuộc nhóm nào)\b",
        r"\b(phân biệt|khác nhau|so sánh)\b",
        r"\b(đặc điểm nhận biết|nhận diện|dấu hiệu)\b",
        r"\b(họ|chi|loài|phân loại|taxonomy)\b",
    ]
    if any(re.search(p, q) for p in definition_signals):
        # ngoại lệ: nếu rõ ràng hỏi nội bộ, để heuristic nội bộ bên dưới override
        pass
    else:
        definition_signals = []  # không dùng

    # ---------------------------
    # 2) Heuristic: tín hiệu nội bộ -> RAG (ưu tiên cao hơn)
    # ---------------------------
    internal_signals = [
        r"\b(bmcvn|bmcvn\.|bmc)\b",
        r"\b(sop|quy trình nội bộ|tài liệu nội bộ|nội bộ)\b",
        r"\b(kb|knowledge base|vector|embedding|tag|tags_v2)\b",
        r"\b(phác đồ|công thức)\b",  # nếu bạn coi "công thức" là nội bộ (tuỳ policy)
        r"\b(mã|code|chunk_|doc\s*\d+)\b",
    ]
    # Nếu có tín hiệu nội bộ mạnh -> RAG ngay
    if any(re.search(p, q) for p in internal_signals):
        return "RAG"

    # Nếu là intent định nghĩa/khái niệm -> GLOBAL
    if definition_signals:
        return "GLOBAL"

    # ---------------------------
    # 3) LLM router (khi heuristic không quyết được)
    # ---------------------------
    system_prompt = """
Bạn là bộ phân luồng câu hỏi cho hệ thống trợ lý nông nghiệp của BMCVN.

NHIỆM VỤ:
- Trả lời đúng 1 từ: GLOBAL hoặc RAG.
- GLOBAL: câu hỏi kiến thức chung/giáo trình (định nghĩa, phân loại, viết tắt, cơ chế chung, khái niệm nông học/BVTV phổ thông).
- RAG: câu hỏi có khả năng phụ thuộc dữ liệu/tài liệu nội bộ BMCVN (tên sản phẩm nội bộ, SOP/phác đồ nội bộ, dữ liệu KB, tag, tài liệu, chunk, mã, bảng, giá/đại lý nội bộ).

LUẬT:
- Chỉ trả lời GLOBAL hoặc RAG, không thêm bất kỳ ký tự nào khác.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    ans = (resp.choices[0].message.content or "").strip().upper()

    # Parse an toàn: chỉ nhận đúng 1 từ
    if ans == "GLOBAL":
        return "GLOBAL"
    if ans == "RAG":
        return "RAG"

    # Fallback: an toàn theo hướng RAG (tránh bịa khi không chắc)
    return "RAG"
