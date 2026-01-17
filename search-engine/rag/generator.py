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
    (2) Nguyên nhân/điều kiện phát sinh
    (3) Triệu chứng
    (4) Hậu quả
    (5) Hướng xử lý & phòng ngừa

    - Quy tắc sử dụng nguồn:
    + Ưu tiên tuyệt đối dữ liệu từ TÀI LIỆU.
    + STRICT mode: chỉ sử dụng thông tin có trong TÀI LIỆU, không bổ sung ngoài.
    + SOFT mode: có thể bổ sung kiến thức chung nhưng không được mâu thuẫn với TÀI LIỆU.

    - Chỉ tạo mục khi TÀI LIỆU có dữ liệu cho mục đó.
    - Không bịa thuốc, liều lượng, thời gian cách ly (TGCL).
    - Chỉ liệt kê thuốc hoặc hoạt chất khi TÀI LIỆU đề cập rõ ràng.
    - Không tự suy diễn thuốc từ tên bệnh.

    - Nếu câu hỏi đề cập nhiều bệnh: trình bày tách riêng từng bệnh theo cùng cấu trúc.

    - Nếu chỉ có một số mục được tạo:
    + Đánh số lại liên tục (1,2,3...) theo thứ tự xuất hiện.
    + Không giữ số gốc (không dùng (5) nếu (2)(3)(4) không tồn tại).

    - Mọi thông tin lấy từ TÀI LIỆU phải được trình bày dưới dạng “Thông tin từ tài liệu”.
    - Nội dung bổ sung (SOFT) phải ghi chú rõ là “Kiến thức chung”.
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
    • Suy diễn mức độ hiệu lực của cơ chế.
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

    ------------------------------------------------------
    PHẦN BỔ SUNG – QUY TẮC XỬ LÝ TRÙNG LẶP & CHUẨN HÓA
    ------------------------------------------------------

    - TRƯỚC KHI trình bày kết quả, BẮT BUỘC thực hiện bước chuẩn hóa danh sách sản phẩm:

    1. PHÁT HIỆN TRÙNG LẶP:
    - Xem các mục có cùng tên thương mại nhưng khác cách ghi (ví dụ có thêm ghi chú trong ngoặc, khác đơn vị %, g/kg, w/w…)
    - Xem các phiên bản cùng sản phẩm nhưng khác cách đặt tên phụ (ví dụ bản quốc gia, bản đóng gói khác)
    → coi là MỘT sản phẩm duy nhất.

    2. GỘP SẢN PHẨM TRÙNG:
    - Giữ một tên chuẩn ngắn gọn nhất.
    - Tổng hợp thông tin hoạt chất một cách thống nhất.
    - Không lặp lại cùng một sản phẩm nhiều lần.

    3. QUY TẮC GỘP:
    - Chỉ gộp khi chắc chắn cùng một sản phẩm.
    - Không gộp nếu:
        • khác cơ chế tác động,
        • khác cây trồng áp dụng,
        • hoặc khác mục đích sử dụng.

    4. ƯU TIÊN TRÌNH BÀY:
    - Nếu query có yêu cầu cụ thể (cây trồng, cơ chế, đối tượng):
        → ưu tiên trình bày các sản phẩm KHỚP HOÀN TOÀN trước.
    - Sản phẩm chỉ khớp một phần → để vào mục "Thông tin tham khảo".

    5. HÌNH THỨC TRÌNH BÀY:
    - Sau khi dedup, mỗi sản phẩm xuất hiện tối đa 1 lần.
    - Tên sản phẩm viết chuẩn, không kèm các hậu tố dư thừa.

    - TUYỆT ĐỐI KHÔNG:
    • Tự thêm sản phẩm ngoài danh sách tài liệu.
    • Tự hợp nhất các sản phẩm khác hoạt chất thành một.
    • Suy diễn rằng hai mục “gần giống tên” là một nếu tài liệu không xác nhận.
    """.strip()

    elif answer_mode == "listing":
        mode_requirements = """
    MỤC TIÊU: LIỆT KÊ SẢN PHẨM TỪ TÀI LIỆU (LISTING MODE)

    Yêu cầu chung:
    - Chỉ liệt kê CÁC SẢN PHẨM thực sự xuất hiện trong TÀI LIỆU được cung cấp.
    - Output phải “SẠCH”: chỉ gồm các dòng sản phẩm hợp lệ.
    - KHÔNG có đoạn giải thích, KHÔNG tổng kết, KHÔNG nhận xét.

    ------------------------------------------------------
    A. QUY TẮC XÁC ĐỊNH INTENT TỪ CÂU HỎI
    ------------------------------------------------------

    1. Nếu câu hỏi KHÔNG đề cập tới cơ chế tác động
    (không chứa các từ khóa: “lưu dẫn”, “tiếp xúc”, “xông hơi”, “nội hấp”, “thấm sâu”…):

    → BỎ QUA hoàn toàn mọi ràng buộc về cơ chế.
    → Chỉ cần sản phẩm thỏa:
        - Đúng đối tượng (bệnh/sâu/cây trồng) theo tài liệu
        - Có tên thương mại rõ ràng trong tài liệu

    2. Chỉ khi câu hỏi CÓ YÊU CẦU CỤ THỂ về cơ chế:
    (ví dụ: “thuốc lưu dẫn”, “cơ chế tiếp xúc”, “xông hơi mạnh”…)

    → Mới áp dụng ràng buộc cơ chế như bên dưới.

    ------------------------------------------------------
    B. RÀNG BUỘC THEO CƠ CHẾ (CHỈ KHI QUERY YÊU CẦU)
    ------------------------------------------------------

    NẾU câu hỏi yêu cầu cơ chế tác động:

    • Chỉ liệt kê sản phẩm mà TÀI LIỆU xác nhận RÕ cơ chế đó,
    dựa trên TAG hoặc mô tả trực tiếp.

    • Thứ tự ưu tiên kiểm tra:
    1) Ưu tiên dùng TAG:
        - mechanism:systemic
        - mechanism:contact
        - mechanism:fume
        …

    2) Nếu không có tag → mới xét mô tả text trong tài liệu.

    • TUYỆT ĐỐI KHÔNG suy diễn:
    - “hiệu quả cao” → KHÔNG đồng nghĩa “lưu dẫn”
    - “diệt nhanh” → KHÔNG đồng nghĩa “tiếp xúc”
    - “thấm nhanh” → KHÔNG suy ra “nội hấp”

    • Nếu tài liệu chỉ xác nhận MỘT phần cơ chế trong khi query yêu cầu NHIỀU cơ chế
    → LOẠI sản phẩm đó.

    ------------------------------------------------------
    C. ĐIỀU KIỆN BẮT BUỘC ĐỂ MỘT SẢN PHẨM ĐƯỢC LIỆT KÊ
    ------------------------------------------------------

    Một sản phẩm CHỈ được liệt kê khi hội đủ:

    (1) Tên thương mại sản phẩm xuất hiện rõ ràng trong tài liệu.
    (2) Tài liệu xác nhận sản phẩm dùng đúng:
        - đối tượng sâu/bệnh/cây trồng mà câu hỏi đề cập.
    (3) (Chỉ khi query yêu cầu) thỏa ràng buộc về cơ chế.

    ------------------------------------------------------
    D. QUY TẮC DEDUP & CHUẨN HÓA
    ------------------------------------------------------

    TRƯỚC KHI TRẢ KẾT QUẢ PHẢI THỰC HIỆN:

    1. LOẠI BỎ TRÙNG LẶP:
    - Nếu cùng một sản phẩm nhưng được ghi nhiều cách:
        • “Tatsu 25WP”
        • “Tatsu 25WP (M8-Singapore)”
        → chỉ giữ MỘT dòng đại diện.

    2. CHỈ GỘP khi:
    - cùng tên thương mại gốc
    - cùng hoạt chất chính
    - cùng mục đích sử dụng

    3. KHÔNG gộp khi:
    - khác hoạt chất
    - khác đối tượng sử dụng
    - khác dạng sản phẩm

    4. CHUẨN HÓA:
    - Bỏ ghi chú phụ không cần thiết trong ngoặc.
    - Dùng cùng một cách ghi hoạt chất cho toàn bộ danh sách.

    ------------------------------------------------------
    E. ĐỊNH DẠNG OUTPUT BẮT BUỘC
    ------------------------------------------------------

    - Mỗi sản phẩm 1 dòng duy nhất.
    - Cấu trúc:

    “TÊN SẢN PHẨM – Hoạt chất: ...”

    Ví dụ:
    Zigen Super 15SC – Hoạt chất: Tolfenpyrad
    Abinsec Oxatin 1.8EC – Hoạt chất: Abamectin

    ------------------------------------------------------
    F. TUYỆT ĐỐI KHÔNG
    ------------------------------------------------------

    • Không liệt kê sản phẩm kèm chú thích kiểu:
    - “không chắc”
    - “có thể”
    - “chưa rõ”

    • Không thêm thông tin ngoài tài liệu.

    • Không tự suy diễn để thêm hoặc loại bỏ sản phẩm.

    • Không tạo đoạn văn giải thích.

    ------------------------------------------------------
    G. NGUYÊN TẮC VÀNG

    CHỈ LIỆT KÊ những gì TÀI LIỆU XÁC NHẬN RÕ RÀNG.
    """.strip()

    elif answer_mode == "procedure":
        mode_requirements = """
- Trình bày theo checklist từng bước.
- Mỗi bước: (Việc cần làm) + (Mục đích) nếu TÀI LIỆU có.
- Không tự phát minh quy trình mới ngoài TÀI LIỆU (STRICT).
- Nếu thiếu bước quan trọng, chỉ được bổ sung dưới dạng "Kiến thức chung" (SOFT) và không kèm số liệu định lượng.
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
    # #dev
    # selected_model = "gpt-4.1-mini"
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
