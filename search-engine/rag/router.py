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
        # ---------------------------
        # 1) Mẫu câu hỏi định nghĩa / giáo trình
        # ---------------------------
        r"(là gì|nghĩa là gì|định nghĩa|khái niệm|hiểu là gì|gì\??)",
        r"\b(bao gồm|gồm những|gồm các|thuộc nhóm nào)\b",
        r"\b(phân biệt|khác nhau|so sánh|giống nhau)\b",
        r"(vì sao|tại sao|nguyên nhân|cơ chế|nguyên lý|sao?)",
        r"\b(ưu điểm|nhược điểm|lợi ích|rủi ro)\b",
        r"\b(nên|không nên|phương án)\b",

        # ---------------------------
        # 2) Nhận diện – triệu chứng – đặc điểm
        # ---------------------------
        r"\b(đặc điểm nhận biết|nhận diện|dấu hiệu|biểu hiện|triệu chứng)\b",
        r"\b(vòng đời|chu kỳ sinh trưởng|giai đoạn sinh trưởng)\b",

        # ---------------------------
        # 3) Phân loại sinh học – taxonomy
        # ---------------------------
        r"\b(họ|chi|loài|phân loại|taxonomy)\b",
        r"\b(nhóm sâu|nhóm bệnh|nhóm dịch hại)\b",
        r"\b(thiên địch|môi trường|tồn dư)\b",

        # ---------------------------
        # 4) Nông học – canh tác (Agronomy)
        # ---------------------------
        r"\b(nông học|canh tác|thâm canh|luân canh|xen canh)\b",
        r"\b(giống cây|giống trồng|lai f1|variety|cultivar)\b",
        r"\b(thời vụ|lịch thời vụ|mật độ trồng|khoảng cách trồng)\b",
        r"\b(gieo hạt|ươm cây|cấy|trồng dặm|tỉa cành|tỉa thưa)\b",
        r"\b(tưới nước|thoát nước|úng|hạn|che phủ)\b",
        r"\b(nhà lưới|nhà kính|greenhouse|giá thể|thủy canh|khí canh)\b",

        # ---------------------------
        # 5) Đất – dinh dưỡng cây trồng
        # ---------------------------
        r"\b(đất trồng|kết cấu đất|độ tơi xốp|độ phì)\b",
        r"\b(pH|EC|độ mặn|salinity|chua|kiềm)\b",
        r"\b(NPK|đạm|lân|kali|trung lượng|vi lượng)\b",
        r"\b(canxi|magiê|lưu huỳnh|sắt|kẽm|bo|đồng|mangan)\b",
        r"\b(bón lót|bón thúc|bón lá|fertigation)\b",
        r"\b(thiếu dinh dưỡng|thừa dinh dưỡng|ngộ độc dinh dưỡng)\b",

        # ---------------------------
        # 6) Sâu hại – dịch hại (Pests)
        # ---------------------------

        r"\b(những bệnh|những biểu hiệu|những sâu|những loại nhện)\b", 
        r"\b(những loại rầy|tác nhân gây hại|tác nhân)\b",

        # ---------------------------
        # 9) Dạng thuốc – formulation (GLOBAL rất mạnh)
        # ---------------------------
        r"\b(EC|SC|WP|WG|WDG|SL|SP|GR|ME|EW|OD|CS|FS)\b",

        # ---------------------------
        # 11) Thu hoạch – sau thu hoạch
        # ---------------------------
        r"\b(thu hoạch|độ chín|bảo quản|sau thu hoạch)\b",
        r"\b(kho lạnh|chuỗi lạnh|nấm mốc kho)\b",
    ]

    # 0) Product / Drug intent -> RAG (ƯU TIÊN CAO NHẤT)
    product_signals = [
        r"(?:^|\s)(thuốc|sản phẩm)(?:\s|$)"
    ]

    for p in product_signals:
        if re.search(p, q):
            print("Matched product signal:", p)
            return "RAG"

    if any(re.search(p, q) for p in product_signals):
        return "RAG"

    # 1) Internal / KB intent -> RAG
    internal_signals = [
        r"\b(bmcvn|bmcvn\.|bmc)\b",
        r"\b(sop|quy trình nội bộ|tài liệu nội bộ|nội bộ)\b",
        r"\b(kb|knowledge base|vector|embedding|tag|tags_v2)\b",
        r"\b(phác đồ)\b",
        r"\b(mã|code|chunk_|doc\s*\d+)\b",
    ]
    if any(re.search(p, q) for p in internal_signals):
        return "RAG"

    # 2) Definition / giáo trình -> GLOBAL
    if any(re.search(p, q) for p in definition_signals):
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
