import re
from rag.debug_log import debug_log

# -----------------------------
# Listing post-filter constants
# -----------------------------

NON_LISTING_PHRASES = [
    "không liệt kê", "nên không liệt kê", "không nêu", "không khuyến khích",
    "được nhắc đến nhưng", "chưa có thông tin xác nhận", "tóm lại", "danh sách", "kết luận",
]

NEG_BAP_SIGNALS = [
    "không được phun trên", "không phun trên", "không dùng cho bắp",
    "ít dùng cho bắp", "gây ngộ độc", "trắng lá"
]

NON_SELECTIVE_HINTS = ["không chọn lọc", "diệt sạch", "diệt mọi loại cỏ"]
NON_SELECTIVE_ACTIVES = ["glufosinate", "glyphosate", "paraquat", "diquat"]
LUA_ONLY_SIGNALS = ["lúa", "lúa sạ", "ruộng lúa", "nước ngập"]


# -----------------------------
# Small helpers
# -----------------------------

def post_filter_product_output(model_text: str) -> str:
    """
    Chỉ giữ các block có '6) Kết luận phù hợp: [PHÙ HỢP]'.
    Nếu không còn block nào, trả câu mặc định.
    """
    text = (model_text or "").strip()
    if not text:
        return "Không có sản phẩm PHÙ HỢP trong tài liệu."

    # Tách theo block sản phẩm: dựa vào "1) Tên sản phẩm:"
    parts = re.split(r"(?=^1\)\s*Tên sản phẩm\s*:)", text, flags=re.MULTILINE)

    kept_blocks = []
    for p in parts:
        blk = p.strip()
        if not blk:
            continue
        # Chỉ giữ block có nhãn PHÙ HỢP
        if re.search(r"^6\)\s*Kết luận phù hợp\s*:\s*\[PHÙ HỢP\]\s*$", blk, flags=re.MULTILINE):
            # Loại phòng trường hợp có nhãn khác lẫn vào
            if ("[KHÔNG PHÙ HỢP]" in blk) or ("[CHƯA XÁC NHẬN]" in blk):
                continue
            kept_blocks.append(blk)

    if not kept_blocks:
        return "Không có sản phẩm PHÙ HỢP trong tài liệu."

    return "\n\n".join(kept_blocks).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _rename_context_terms(text: str) -> str:
    """
    Đổi cách gọi NGỮ CẢNH -> TÀI LIỆU để dễ nghe hơn với người dùng cuối.
    Lưu ý: dùng replace đơn giản theo đúng casing bạn đang dùng trong prompt.
    """
    if not text:
        return text

    # Ưu tiên thay "Không thấy trong ngữ cảnh" trước để không tạo câu lai
    text = text.replace("Không thấy trong ngữ cảnh", "Không thấy trong tài liệu")
    text = text.replace("không thấy trong ngữ cảnh", "không thấy trong tài liệu")

    # Thay các cụm chính
    text = text.replace("NGỮ CẢNH", "TÀI LIỆU")
    text = text.replace("ngữ cảnh", "tài liệu")

    return text


# -----------------------------
# Model selection
# -----------------------------

def select_model_for_query(user_query: str, answer_mode: str, any_tags=None) -> str:
    q = (user_query or "").lower()
    any_tags = any_tags or []

    HIGH_RISK_TAGS = {
        "mechanisms:co-chon-loc",
        "mechanisms:herbicide",   # nếu anh có
        "crop:bap", "crop:ngo", "crop:mia", "crop:lua",
    }

    HIGH_RISK_KEYWORDS = [
        "chọn lọc", "thuốc trừ cỏ", "thuốc cỏ",
        "bắp", "ngô", "mía", "lúa",
    ]

    # 1) Nếu router đã gắn tag rủi ro → 4.1
    if any(t in HIGH_RISK_TAGS for t in any_tags):
        return "gpt-4.1"

    # 2) Nếu query có keyword rủi ro → 4.1
    if any(kw in q for kw in HIGH_RISK_KEYWORDS):
        return "gpt-4.1"

    # 3) Default
    return "gpt-4.1-mini"


# -----------------------------
# Listing output post-filter
# -----------------------------

def post_filter_listing_output(model_text: str, user_query: str, any_tags=None) -> str:
    any_tags = any_tags or []
    q = _norm(user_query)

    require_selective = ("chọn lọc" in q) or ("mechanisms:co-chon-loc" in any_tags)
    require_bap = ("crop:bap" in any_tags) or ("bắp" in q) or ("ngô" in q)

    lines = [ln.strip() for ln in (model_text or "").splitlines() if ln.strip()]
    kept = []

    for ln in lines:
        ln_norm = _norm(ln)

        # 1) bỏ mọi dòng “listing bẩn”
        if any(p in ln_norm for p in NON_LISTING_PHRASES):
            continue
        if ln_norm.startswith(("tóm lại", "danh sách", "kết luận")):
            continue

        # 2) nếu query yêu cầu chọn lọc → loại thuốc không chọn lọc
        if require_selective:
            if any(h in ln_norm for h in NON_SELECTIVE_HINTS):
                continue
            if any(a in ln_norm for a in NON_SELECTIVE_ACTIVES):
                continue

        # 3) nếu query là bắp → loại tín hiệu “lúa-only”
        if require_bap:
            if any(sig in ln_norm for sig in LUA_ONLY_SIGNALS):
                continue

        kept.append(ln)

    # 4) loại trùng theo dòng
    seen = set()
    unique = []
    for ln in kept:
        key = _norm(re.sub(r"[-–—]+", "-", ln))
        if key in seen:
            continue
        seen.add(key)
        unique.append(ln)

    return "\n".join(unique).strip()


# -----------------------------
# Main: call finetune/chat with context
# -----------------------------

def call_finetune_with_context(
    client,
    user_query: str,
    context: str,
    answer_mode: str = "general",
    rag_mode: str = "STRICT",
    must_tags=None,
    any_tags=None,
):
    must_tags = must_tags or []
    any_tags = any_tags or []

    print("answer_mode:", answer_mode)

    BASE_REASONING_PROMPT = """
Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN.

GHI CHÚ QUAN TRỌNG:
- "TÀI LIỆU" ở đây là phần văn bản được cung cấp trong prompt (không phải nguồn bên ngoài).
- Không được dùng bất kỳ nguồn ngoài nào; chỉ dựa vào TÀI LIỆU.

NGUYÊN TẮC BẮT BUỘC:
1) Ưu tiên TÀI LIỆU. Chỉ dùng thông tin có trong TÀI LIỆU cho các dữ liệu định lượng/chỉ định chi tiết như:
   - liều lượng, cách pha, lượng nước, thời gian cách ly, tần suất phun, nồng độ, khuyến cáo kỹ thuật cụ thể.
2) Nếu cần bổ sung kiến thức phổ biến để giải thích mạch lạc (không phải số liệu/khuyến cáo định lượng), có thể bổ sung ở mức "kiến thức chung"
   và phải dùng các cụm: "Thông tin chung:", "Thông lệ kỹ thuật:".
   Hoặc các câu hỏi liên quan đến thông tin khoa về sâu hại, bệnh hại, vụ mùa.
3) Tuyệt đối không bịa. Nếu TÀI LIỆU không có, hãy để trống/ghi "Không thấy trong tài liệu" thay vì suy đoán.
4) Mục tiêu: câu trả lời hữu ích cho nhân viên/khách hàng, có cấu trúc, đầy đủ, dễ so sánh.
5) Nếu câu hỏi liên quan thủy sinh (cá/tôm/vật nuôi...), mà TÀI LIỆU không đề cập: phải nhấn mạnh "Tài liệu không đề cập".

6) CHỐNG SUY DIỄN PHẠM VI (BẮT BUỘC):
   - TUYỆT ĐỐI KHÔNG suy luận mở rộng phạm vi sử dụng thuốc từ cây trồng A sang cây trồng B dựa trên:
     • loại cỏ/sâu/bệnh tương tự,
     • cơ chế tác động,
     • hoặc các câu kiểu “áp dụng gián tiếp”, “dùng tương tự”, “tham khảo”.
   - Chỉ coi là PHÙ HỢP khi TÀI LIỆU xác nhận rõ cây trồng/phạm vi/đối tượng sử dụng KHỚP với câu hỏi.

7) QUY TẮC AN TOÀN THUỐC CỎ:
   - Nếu TÀI LIỆU không xác nhận rõ cây trồng/phạm vi dùng được, thì KHÔNG được gợi ý/đề xuất sử dụng.
   - Đặc biệt cảnh giác với mô tả “không chọn lọc/diệt sạch/diệt mọi loại cỏ”: nếu thiếu xác nhận phù hợp với câu hỏi → không khuyến nghị, không suy diễn.

YÊU CẦU TRÌNH BÀY:
- Không tối ưu cho ngắn gọn.
- Ưu tiên tính đúng, đầy đủ, nhất quán.
""".strip()

    # -----------------------------
    # Mode requirements
    # -----------------------------
    if answer_mode == "disease":
        mode_requirements = """
- Cấu trúc ưu tiên:
  (1) Tổng quan
  (2) Nguyên nhân/điều kiện phát sinh (chỉ khi có trong TÀI LIỆU; SOFT có thể bổ sung kiến thức chung)
  (3) Triệu chứng (chỉ khi có)
  (4) Hậu quả (chỉ khi có)
  (5) Hướng xử lý & phòng ngừa (ưu tiên biện pháp trong TÀI LIỆU)
- Không tạo mục nếu TÀI LIỆU không có dữ liệu cho mục đó.
- Không bịa thuốc/liều/TGCL.
- Nếu chỉ có một số mục được tạo, hãy ĐÁNH SỐ LẠI LIÊN TỤC (1,2,3...) theo thứ tự xuất hiện.
- Không giữ số gốc (ví dụ không dùng (5) nếu (2)(3)(4) không tồn tại).
""".strip()

    elif answer_mode == "product":
        mode_requirements = """
    MODE: PRODUCT (EVIDENCE-ONLY, QUERY-CONDITIONED)

    - Trình bày chi tiết, không trả lời quá ngắn gọn.
    - Mỗi sản phẩm phải được trình bày TÁCH RIÊNG, dựa HOÀN TOÀN vào dữ liệu trong TÀI LIỆU.

    - CHỈ coi một sản phẩm là "ĐỀ XUẤT PHÙ HỢP" khi TÀI LIỆU NÊU RÕ ĐỒNG THỜI:
    • đối tượng trừ (cỏ/sâu/bệnh cụ thể)
    • và cây trồng / phạm vi sử dụng PHÙ HỢP với câu hỏi người dùng
    • và cơ chế tác động KHỚP với yêu cầu trong câu hỏi (nếu có).

    - QUY TẮC CƠ CHẾ TÁC ĐỘNG (BẮT BUỘC):
    • Nếu câu hỏi yêu cầu "lưu dẫn":
        - CHỈ chấp nhận sản phẩm mà TÀI LIỆU xác nhận có cơ chế "lưu dẫn" hoặc "nội hấp".
        - TUYỆT ĐỐI KHÔNG chấp nhận các mô tả suy diễn như "lưu dẫn mạnh", "lưu dẫn tốt", "hiệu quả cao".
    • Nếu câu hỏi yêu cầu "tiếp xúc":
        - CHỈ chấp nhận sản phẩm mà TÀI LIỆU xác nhận cơ chế "tiếp xúc".
    • Nếu câu hỏi yêu cầu "xông hơi":
        - CHỈ chấp nhận sản phẩm mà TÀI LIỆU xác nhận cơ chế "xông hơi".
    • Nếu câu hỏi yêu cầu kết hợp (ví dụ: "tiếp xúc, lưu dẫn"):
        - CHỈ chấp nhận sản phẩm mà TÀI LIỆU xác nhận RÕ CẢ HAI cơ chế.
        - Nếu TÀI LIỆU chỉ nêu 1 trong 2 → KHÔNG coi là phù hợp.
    - TUYỆT ĐỐI KHÔNG:
    • Suy diễn mức độ hiệu lực của cơ chế (không dùng các từ như: mạnh, yếu, cao, tốt).
    • Diễn giải lại cơ chế theo ý hiểu nếu TÀI LIỆU không nêu.
    • Mở rộng cơ chế từ "tiếp xúc" sang "lưu dẫn" hoặc ngược lại.

    - Nếu sản phẩm chỉ được mô tả chung (ví dụ: "trừ sâu phổ rộng"),
    nhưng TÀI LIỆU KHÔNG NÊU RÕ cơ chế / đối tượng / cây trồng đang hỏi
    → KHÔNG ĐƯỢC đưa vào phần đề xuất hay khuyến nghị.

    - ĐƯỢC PHÉP:
    • Mô tả sản phẩm đó như thông tin tham khảo
    • nhưng BẮT BUỘC phải ghi rõ: "chưa có thông tin xác nhận về cơ chế ... dùng cho ..."

    - Không tự bịa thêm liều lượng, cách pha, thời gian cách ly.
    - Không tổng hợp hoặc gộp sản phẩm nếu điều kiện sử dụng khác nhau.
    """.strip()

    elif answer_mode == "procedure":
        mode_requirements = """
- Trình bày theo checklist từng bước.
- Mỗi bước: (Việc cần làm) + (Mục đích) nếu TÀI LIỆU có.
- Không tự phát minh quy trình mới ngoài TÀI LIỆU (STRICT).
- Nếu thiếu bước quan trọng, chỉ được bổ sung dưới dạng "Kiến thức chung" (SOFT) và không kèm số liệu định lượng.
""".strip()

    elif answer_mode == "listing":
        mode_requirements = """
- Mục tiêu: LIỆT KÊ các SẢN PHẨM xuất hiện trong TÀI LIỆU.
Output phải "SẠCH": chỉ gồm các dòng sản phẩm hợp lệ. KHÔNG có phần giải thích, KHÔNG có tổng kết dạng văn.

- RÀNG BUỘC THEO CÂU HỎI (QUERY-CONDITIONED – BẮT BUỘC):
  • Nếu câu hỏi yêu cầu cơ chế tác động (ví dụ: "lưu dẫn", "tiếp xúc", "xông hơi"):
      - CHỈ liệt kê sản phẩm mà TÀI LIỆU xác nhận ĐÚNG cơ chế đó.
      - TUYỆT ĐỐI KHÔNG suy diễn hoặc chấp nhận các mô tả như:
        "lưu dẫn mạnh", "tác động tốt", "hiệu quả cao", "gần giống lưu dẫn".

  • Nếu câu hỏi yêu cầu cơ chế kết hợp (ví dụ: "tiếp xúc, lưu dẫn"):
      - CHỈ liệt kê sản phẩm mà TÀI LIỆU xác nhận RÕ CẢ HAI cơ chế.
      - Nếu TÀI LIỆU chỉ nêu một cơ chế → LOẠI.

- ĐIỀU KIỆN BẮT BUỘC (HARD GATE) ĐỂ LIỆT KÊ 1 SẢN PHẨM:
  (1) Tên thương mại sản phẩm xuất hiện trong TÀI LIỆU.
  (2) TÀI LIỆU xác nhận cây trồng/phạm vi sử dụng KHỚP với câu hỏi.
  (3) TÀI LIỆU xác nhận ĐÚNG cơ chế tác động mà câu hỏi yêu cầu (nếu có).

- CHỐNG SUY DIỄN (RẤT QUAN TRỌNG):
  • TUYỆT ĐỐI KHÔNG suy luận:
      - "có hiệu quả cao" → "lưu dẫn"
      - "diệt nhanh" → "tiếp xúc"
      - "thấm nhanh" → "lưu dẫn"
  • Chỉ chấp nhận đúng thuật ngữ cơ chế xuất hiện trong TÀI LIỆU.

- TUYỆT ĐỐI KHÔNG:
  • Không liệt kê sản phẩm rồi chú thích "không chắc", "chưa rõ".
  • Không dùng các cụm từ so sánh mức độ (mạnh/yếu/tốt/kém).
  • Không có đoạn giải thích hay tổng kết.

- Định dạng output:
  • Mỗi sản phẩm 1 dòng.
  • Dạng: "TÊN SẢN PHẨM – Hoạt chất: ...".
""".strip()

    else:
        mode_requirements = """
- Trình bày có cấu trúc theo ý chính.
- Ưu tiên tổng hợp từ nhiều đoạn TÀI LIỆU.
- Không bịa số liệu/liều lượng nếu TÀI LIỆU không có.
- Nếu câu hỏi liên quan thủy sinh (cá/tôm/vật nuôi...), mà TÀI LIỆU không đề cập: phải nhấn mạnh "Tài liệu không đề cập".
""".strip()

    # -----------------------------
    # RAG STRICT vs SOFT
    # -----------------------------
    if rag_mode == "SOFT":
        system_prompt = (
            BASE_REASONING_PROMPT
            + "\nSOFT MODE: chỉ được bổ sung 'kiến thức chung' để GIẢI THÍCH KHÁI NIỆM/CƠ CHẾ cho mạch lạc."
            + "\nSOFT MODE KHÔNG cho phép: suy luận mở rộng phạm vi dùng thuốc sang cây trồng khác; KHÔNG được tạo khuyến cáo sử dụng nếu TÀI LIỆU không xác nhận."
            + "\nSOFT MODE vẫn cấm: số liệu/liều/TGCL/cách pha nếu TÀI LIỆU không có."
        )
    else:
        system_prompt = (
            BASE_REASONING_PROMPT
            + "\nSTRICT MODE: chỉ dùng TÀI LIỆU. Không thêm kiến thức ngoài, không ngoại suy phạm vi sử dụng."
            + "\nChỉ được diễn giải lại cho dễ hiểu, nhưng KHÔNG tạo kết luận mới ngoài TÀI LIỆU."
        )

    # -----------------------------
    # User prompt (uses "TÀI LIỆU")
    # -----------------------------
    user_prompt = f"""
TÀI LIỆU (chỉ được dùng các dữ kiện định lượng từ đây):
\"\"\"{context}\"\"\"

CÂU HỎI:
\"\"\"{user_query}\"\"\"
MUST TAGS  : {must_tags}
ANY TAGS   : {any_tags}
MODE: {answer_mode}

CHỈ THỊ CHUNG (bắt buộc):
- Không bịa số liệu/liều lượng/cách pha/TGCL nếu TÀI LIỆU không nêu.
- Nếu có thể, ưu tiên tổng hợp từ nhiều đoạn TÀI LIỆU (không chỉ 1–2 đoạn).
- Khi liệt kê sản phẩm/tên thuốc: tên đó phải xuất hiện trong TÀI LIỆU.
- Không được suy luận “diệt cỏ/sâu/bệnh X” → “dùng được cho cây trồng Y” nếu TÀI LIỆU không xác nhận.
- Với thuốc cỏ: chỉ coi là phù hợp khi TÀI LIỆU nêu rõ cây trồng/phạm vi sử dụng khớp câu hỏi.

CHỈ THỊ RIÊNG THEO MODE:
{mode_requirements}
""".strip()

    selected_model = select_model_for_query(
        user_query=user_query,
        answer_mode=answer_mode,
        any_tags=any_tags,
    )
    debug_log(selected_model)

    resp = client.chat.completions.create(
        model=selected_model,
        temperature=0.4,
        max_completion_tokens=3500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    # if answer_mode == "product":
    #     raw = post_filter_product_output(raw)

    # Listing post-filter
    if answer_mode == "listing":
        raw = post_filter_listing_output(
            model_text=raw,
            user_query=user_query,
            any_tags=any_tags,
        )

    return raw
