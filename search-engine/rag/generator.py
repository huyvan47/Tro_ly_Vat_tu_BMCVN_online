import re

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

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def post_filter_listing_output(model_text: str, user_query: str, any_tags=None) -> str:
    any_tags = any_tags or []
    q = _norm(user_query)

    require_selective = ("chọn lọc" in q) or ("mechanisms:co-chon-loc" in any_tags)
    require_bap = ("crop:bap" in any_tags) or ("bắp" in q) or ("ngô" in q)

    lines = [ln.strip() for ln in model_text.splitlines() if ln.strip()]
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


def call_finetune_with_context(
    client,
    user_query: str,
    context: str,
    answer_mode: str = "general",
    rag_mode: str = "STRICT",
):
    print("answer_mode:", answer_mode)

    BASE_REASONING_PROMPT = """
Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN.

NGUYÊN TẮC BẮT BUỘC:
1) Ưu tiên NGỮ CẢNH. Chỉ dùng thông tin có trong NGỮ CẢNH cho các dữ liệu định lượng/chỉ định chi tiết như:
   - liều lượng, cách pha, lượng nước, thời gian cách ly, tần suất phun, nồng độ, khuyến cáo kỹ thuật cụ thể.
2) Nếu cần bổ sung kiến thức phổ biến để giải thích mạch lạc (không phải số liệu/khuyến cáo định lượng), có thể bổ sung ở mức "kiến thức chung"
   và phải dùng các cụm: "Thông tin chung:", "Thông lệ kỹ thuật:".
   Hoặc các câu hỏi liên quan đến thông tin khoa về sâu hại, bệnh hại, vụ mùa.
3) Tuyệt đối không bịa. Nếu NGỮ CẢNH không có, hãy để trống/ghi "Không thấy trong ngữ cảnh" thay vì suy đoán.
4) Mục tiêu: câu trả lời hữu ích cho nhân viên/khách hàng, có cấu trúc, đầy đủ, dễ so sánh.
5) Nếu câu hỏi liên quan thủy sinh (cá/tôm/vật nuôi...), mà NGỮ CẢNH không đề cập: phải nhấn mạnh "Tài liệu không đề cập".

6) CHỐNG SUY DIỄN PHẠM VI (BẮT BUỘC):
   - TUYỆT ĐỐI KHÔNG suy luận mở rộng phạm vi sử dụng thuốc từ cây trồng A sang cây trồng B dựa trên:
     • loại cỏ/sâu/bệnh tương tự,
     • cơ chế tác động,
     • hoặc các câu kiểu “áp dụng gián tiếp”, “dùng tương tự”, “tham khảo”.
   - Chỉ coi là PHÙ HỢP khi NGỮ CẢNH xác nhận rõ cây trồng/phạm vi/đối tượng sử dụng KHỚP với câu hỏi.

7) QUY TẮC AN TOÀN THUỐC CỎ:
   - Nếu NGỮ CẢNH không xác nhận rõ cây trồng/phạm vi dùng được, thì KHÔNG được gợi ý/đề xuất sử dụng.
   - Đặc biệt cảnh giác với mô tả “không chọn lọc/diệt sạch/diệt mọi loại cỏ”: nếu thiếu xác nhận phù hợp với câu hỏi → không khuyến nghị, không suy diễn.

YÊU CẦU TRÌNH BÀY:
- Không tối ưu cho ngắn gọn.
- Ưu tiên tính đúng, đầy đủ, nhất quán.
""".strip()

    # Mode requirements
    if answer_mode == "disease":
        mode_requirements = """
- Cấu trúc ưu tiên:
  (1) Tổng quan
  (2) Nguyên nhân/điều kiện phát sinh (chỉ khi có trong NGỮ CẢNH; SOFT có thể bổ sung kiến thức chung)
  (3) Triệu chứng (chỉ khi có)
  (4) Hậu quả (chỉ khi có)
  (5) Hướng xử lý & phòng ngừa (ưu tiên biện pháp trong NGỮ CẢNH)
- Không tạo mục nếu NGỮ CẢNH không có dữ liệu cho mục đó.
- Không bịa thuốc/liều/TGCL.
- Nếu chỉ có một số mục được tạo, hãy ĐÁNH SỐ LẠI LIÊN TỤC (1,2,3...) theo thứ tự xuất hiện.
- Không giữ số gốc (ví dụ không dùng (5) nếu (2)(3)(4) không tồn tại).
""".strip()

    elif answer_mode == "product":
        mode_requirements = """
- Trình bày chi tiết, không trả lời quá ngắn gọn.
- Mỗi sản phẩm phải được trình bày TÁCH RIÊNG, dựa HOÀN TOÀN vào dữ liệu trong NGỮ CẢNH.

- CHỈ coi một sản phẩm là "ĐỀ XUẤT PHÙ HỢP" khi NGỮ CẢNH NÊU RÕ:
  • đối tượng trừ (cỏ/sâu/bệnh cụ thể)
  • và cây trồng / phạm vi sử dụng PHÙ HỢP với câu hỏi của người dùng.

- Nếu một sản phẩm chỉ được mô tả chung (ví dụ: trừ cỏ trên cây trồng cạn),
  nhưng NGỮ CẢNH KHÔNG NÊU RÕ áp dụng cho đối tượng/cây trồng đang hỏi
  → KHÔNG ĐƯỢC đưa vào phần đề xuất, tổng kết hay khuyến nghị sử dụng.

- ĐƯỢC PHÉP:
  • mô tả sản phẩm đó như thông tin tham khảo (khi sản phẩm xuất hiện trong NGỮ CẢNH)
  • nhưng BẮT BUỘC phải ghi rõ: "chưa có thông tin xác nhận dùng cho ..."

- TUYỆT ĐỐI KHÔNG:
  • suy diễn phạm vi sử dụng
  • mở rộng sang cây trồng khác
  • gợi ý sử dụng ngoài những gì NGỮ CẢNH xác nhận.

- Không tự bịa thêm liều lượng, cách pha, thời gian cách ly.
- Không tổng hợp hoặc gộp sản phẩm nếu điều kiện sử dụng khác nhau.
""".strip()

    elif answer_mode == "procedure":
        mode_requirements = """
- Trình bày theo checklist từng bước.
- Mỗi bước: (Việc cần làm) + (Mục đích) nếu NGỮ CẢNH có.
- Không tự phát minh quy trình mới ngoài NGỮ CẢNH (STRICT).
- Nếu thiếu bước quan trọng, chỉ được bổ sung dưới dạng "Kiến thức chung" (SOFT) và không kèm số liệu định lượng.
""".strip()

    elif answer_mode == "listing":
        mode_requirements = """
    - OUTPUT PHẢI LÀ LISTING SẠCH:
    • Chỉ gồm các DÒNG SẢN PHẨM hợp lệ.
    • KHÔNG giải thích, KHÔNG tổng kết, KHÔNG viết “không liệt kê”.

    - HARD GATE BẮT BUỘC (KHÓA CỨNG):
    Một sản phẩm CHỈ ĐƯỢC LIỆT KÊ khi ĐỒNG THỜI thỏa mãn TẤT CẢ điều kiện sau:

    (1) TÊN SẢN PHẨM phải xuất hiện TRỰC TIẾP trong NGỮ CẢNH.
    (2) NGỮ CẢNH xác nhận rõ CÂY TRỒNG / PHẠM VI sử dụng KHỚP với câu hỏi.
        • Nếu câu hỏi là bắp/ngô → ngữ cảnh PHẢI nêu bắp/ngô.
        • Nếu ngữ cảnh chỉ nêu lúa/lúa sạ/ruộng lúa → TUYỆT ĐỐI KHÔNG LIỆT KÊ.
    (3) Nếu câu hỏi có “chọn lọc” (hoặc tag mechanisms:co-chon-loc):
        • NGỮ CẢNH PHẢI xác nhận “chọn lọc” / “tác động chọn lọc” /
            “an toàn chọn lọc trên cây trồng đó”.
        • TUYỆT ĐỐI LOẠI các thuốc được mô tả là:
            “không chọn lọc”, “diệt sạch”, “diệt mọi loại cỏ”.

    - TUYỆT ĐỐI KHÔNG:
    • Không suy luận “diệt cỏ X → dùng được cho cây trồng Y”.
    • Không dùng các câu “áp dụng gián tiếp”, “dùng tương tự”, “tham khảo”.
    • Không viết các câu:
        – “không nêu rõ nên không liệt kê”
        – “không dùng cho … nên không liệt kê”
        – “được nhắc đến nhưng …”

    → KHÔNG ĐỦ ĐIỀU KIỆN = BỎ QUA HOÀN TOÀN, KHÔNG NÊU.

    - CHUẨN HOÁ TÊN:
    • Không phân biệt hoa/thường.
    • Gộp các tên thương mại trùng (ví dụ: “Giáo Sư Cỏ” và “Misung 15SC”).
    • Mỗi sản phẩm chỉ xuất hiện 1 dòng duy nhất.

    - ĐỊNH DẠNG OUTPUT:
    • Mỗi sản phẩm 1 dòng.
    • Dạng: “TÊN SẢN PHẨM – Hoạt chất: …”.
    """.strip()
    
    else:
        mode_requirements = """
- Trình bày có cấu trúc theo ý chính.
- Ưu tiên tổng hợp từ nhiều đoạn NGỮ CẢNH.
- Không bịa số liệu/liều lượng nếu NGỮ CẢNH không có.
- Nếu câu hỏi liên quan thủy sinh (cá/tôm/vật nuôi...), mà NGỮ CẢNH không đề cập: phải nhấn mạnh "Tài liệu không đề cập".
""".strip()

    # RAG STRICT vs SOFT
    if rag_mode == "SOFT":
        system_prompt = (
            BASE_REASONING_PROMPT
            + "\nSOFT MODE: chỉ được bổ sung 'kiến thức chung' để GIẢI THÍCH KHÁI NIỆM/CƠ CHẾ cho mạch lạc."
            + "\nSOFT MODE KHÔNG cho phép: suy luận mở rộng phạm vi dùng thuốc sang cây trồng khác; KHÔNG được tạo khuyến cáo sử dụng nếu NGỮ CẢNH không xác nhận."
            + "\nSOFT MODE vẫn cấm: số liệu/liều/TGCL/cách pha nếu NGỮ CẢNH không có."
        )
    else:
        system_prompt = (
            BASE_REASONING_PROMPT
            + "\nSTRICT MODE: chỉ dùng NGỮ CẢNH. Không thêm kiến thức ngoài, không ngoại suy phạm vi sử dụng."
            + "\nChỉ được diễn giải lại cho dễ hiểu, nhưng KHÔNG tạo kết luận mới ngoài NGỮ CẢNH."
        )

    user_prompt = f"""
NGỮ CẢNH (chỉ được dùng các dữ kiện định lượng từ đây):
\"\"\"{context}\"\"\"

CÂU HỎI:
\"\"\"{user_query}\"\"\"

MODE: {answer_mode}

CHỈ THỊ CHUNG (bắt buộc):
- Không bịa số liệu/liều lượng/cách pha/TGCL nếu NGỮ CẢNH không nêu.
- Nếu có thể, ưu tiên tổng hợp từ nhiều đoạn NGỮ CẢNH (không chỉ 1–2 đoạn).
- Khi liệt kê sản phẩm/tên thuốc: tên đó phải xuất hiện trong NGỮ CẢNH.
- Không được suy luận “diệt cỏ/sâu/bệnh X” → “dùng được cho cây trồng Y” nếu NGỮ CẢNH không xác nhận.
- Với thuốc cỏ: chỉ coi là phù hợp khi NGỮ CẢNH nêu rõ cây trồng/phạm vi sử dụng khớp câu hỏi.

CHỈ THỊ RIÊNG THEO MODE:
{mode_requirements}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        max_completion_tokens=3500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content.strip()

    # Nếu anh/chị có any_tags từ router thì truyền vào đây.
    # Tạm thời không có biến any_tags trong hàm, nên để [].
    if answer_mode == "listing":
        raw = post_filter_listing_output(
            model_text=raw,
            user_query=user_query,
            any_tags=[],
        )

    return raw
