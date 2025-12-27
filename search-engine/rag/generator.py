def call_finetune_with_context(client, user_query, context, answer_mode: str = "general", rag_mode: str = "STRICT"):
    print('answer_mode:', answer_mode)
    # Mode requirements (giữ nguyên tinh thần code v4 của anh)
    BASE_REASONING_PROMPT = """
    Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN.
    NGUYÊN TẮC BẮT BUỘC:
    1) Ưu tiên NGỮ CẢNH. Chỉ dùng thông tin có trong NGỮ CẢNH cho các dữ liệu định lượng/chỉ định chi tiết như:
    - liều lượng, cách pha, lượng nước, thời gian cách ly, tần suất phun, nồng độ, khuyến cáo kỹ thuật cụ thể.
    2) Nếu cần bổ sung kiến thức phổ biến để giải thích mạch lạc (không phải số liệu/khuyến cáo định lượng), có thể bổ sung ở mức "kiến thức chung"
    và phải dùng các cụm: "Thông tin chung:", "Thông lệ kỹ thuật:". Hoặc các câu hỏi liên quan đến thông tin khoa về sâu hại, bệnh hại, vụ mùa.
    3) Tuyệt đối không bịa. Nếu NGỮ CẢNH không có, hãy để trống/ghi "Không thấy trong ngữ cảnh" thay vì suy đoán.
    4) Mục tiêu: câu trả lời hữu ích cho nhân viên/khách hàng, có cấu trúc, đầy đủ, dễ so sánh.
    5) Nếu câu hỏi liên quan thủy sinh (cá/tôm/vật nuôi...), mà NGỮ CẢNH không đề cập: phải nhấn mạnh "Tài liệu không đề cập".
    YÊU CẦU TRÌNH BÀY:
    - Không tối ưu cho ngắn gọn.
    - Ưu tiên tính đúng, đầy đủ, nhất quán.
    """.strip()

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
""".strip()
    elif answer_mode == "product":
        mode_requirements = """
- Trình bày chi tiết, không trả lời quá ngắn gọn.
- Cố gắn trình bày đầy đủ thông tin sản phẩm, lưu ý, công thức từ các DOC được đề xuất (mục tiêu từ 3 - 5  sản phẩm nếu có).
- Làm rõ đặc tính sản phẩm, cơ chế (nếu NGỮ CẢNH có), phạm vi tác động.
- CHỈ sử dụng dữ liệu trong NGỮ CẢNH, không được tự bịa thêm liều lượng, cách pha, thời gian cách ly.
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
- Mục tiêu: tổng hợp đầy đủ các mục xuất hiện trong NGỮ CẢNH, không bỏ sót.
- Trình bày theo nhóm nếu có thể (theo chủ đề/tags/đối tượng).
- Không bịa thêm ngoài NGỮ CẢNH.
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
        system_prompt = ( BASE_REASONING_PROMPT + "\nSOFT MODE: được phép bổ sung 'kiến thức chung' để giải thích mạch lạc, nhưng không đưa số liệu/liều/TGCL nếu NGỮ CẢNH không có.")
    else:
        system_prompt = (BASE_REASONING_PROMPT + "\nSTRICT MODE: chỉ dùng NGỮ CẢNH. Không thêm kiến thức ngoài, trừ diễn giải lại cho dễ hiểu.")



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

    CHỈ THỊ RIÊNG THEO MODE:
    {mode_requirements}
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        max_completion_tokens=3500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return resp.choices[0].message.content.strip()