import re
from .chemical_knowledge import CHEMICAL_REGEX


def route_query(client, user_query: str) -> str:
    """
    Router tối ưu cho nghiệp vụ BMCVN:

    - Nhắc đến tên công ty/nhà cung cấp -> RAG
    - Khi thấy "thuốc" -> hiểu là sản phẩm BMC -> RAG
    - Nhưng ưu tiên GLOBAL cho các câu hỏi kiến thức hoạt chất
    """

    q = (user_query or "").strip().lower()

    # ============================================
    # 0) Ưu tiên cao nhất: nếu nhắc đến công ty -> RAG
    # ============================================

    company_signals = [
        r"\bbmc\b",
        r"\bphúc thịnh\b",
        r"\bphuc thinh\b",
        r"\bdelta\b",
        r"\bagrishop\b",
        r"\bagri shop\b",
        r"\bcông ty\b",
        r"\bcty\b"
    ]

    if any(re.search(p, q) for p in company_signals):
        return "RAG"

    # ============================================
    # 1) Ngoại lệ quan trọng:
    #    Hỏi kiến thức thuần về hoạt chất -> GLOBAL
    # ============================================

    if CHEMICAL_REGEX.search(q):

        product_intent = re.search(
            r"\b("
            r"giá|mua|mã|đại lý|bán ở đâu|"
            r"có trong sản phẩm|"
            r"có trong các sản phẩm|"
            r"sản phẩm nào chứa|"
            r"thuốc nào chứa|"
            r"thuốc có chứa|"
            r"thuốc có|"
            r"thuốc chứa|"
            r"sản phẩm chứa|"
            r"chứa hoạt chất"
            r")\b",
            q
        )

        # Nếu không có dấu hiệu hỏi sản phẩm -> GLOBAL
        if not product_intent:
            return "GLOBAL"

    # ============================================
    # 2) Nếu câu hỏi có từ “thuốc” -> RAG
    # ============================================

    product_context = re.search(
        r"\b(thuốc|sản phẩm|mã|giá|đại lý|mua)\b",
        q
    )

    if product_context:
        return "RAG"

    # ============================================
    # 3) Các câu hỏi mang tính giáo trình -> GLOBAL
    # ============================================

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

    if any(re.search(p, q) for p in definition_signals):
        return "GLOBAL"

    # ============================================
    # 4) Nếu chưa quyết được -> hỏi LLM router
    # ============================================

    system_prompt = """
Bạn là bộ phân luồng câu hỏi cho hệ thống trợ lý nông nghiệp BMCVN.

NHIỆM VỤ:
- Trả lời đúng 1 từ: GLOBAL hoặc RAG.

QUY TẮC:
- GLOBAL: câu hỏi kiến thức chung, giáo trình, định nghĩa, cơ chế, giải thích về hoạt chất, sâu bệnh, cây trồng.
- RAG: câu hỏi về sản phẩm/thuốc của BMC, giá, mã, đại lý, phác đồ nội bộ, tài liệu nội bộ.

LUẬT:
- Chỉ trả lời GLOBAL hoặc RAG, không thêm bất kỳ ký tự nào khác.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )

        ans = (resp.choices[0].message.content or "").strip().upper()

        if ans in ["GLOBAL", "RAG"]:
            return ans

    except Exception:
        pass

    # ============================================
    # 5) Fallback an toàn
    # ============================================

    return "RAG"
