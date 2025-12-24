def call_finetune_with_context(client, user_query, context, answer_mode: str = "general", rag_mode: str = "STRICT"):
    print('answer_mode:', answer_mode)
    # Mode requirements (giữ nguyên tinh thần code v4 của anh)

    BASE_REASONING_PROMPT = """
    You may take as much space as needed to provide the highest-quality answer.
    Do not optimize for brevity.
    Optimize for factual correctness, completeness, internal consistency, and real-world applicability.
    Use careful reasoning and domain knowledge to infer missing but necessary details when appropriate,
    and clearly state assumptions when you do so.
    The goal is to equip customer support staff with the most reliable and comprehensive information possible.
    """.strip()

    if answer_mode == "disease":
        mode_requirements = """
- Trình bày chi tiết, dễ hiểu cho người trồng. Không trả lời kiểu 1–2 câu là xong.
- Nếu NGỮ CẢNH không mô tả chi tiết triệu chứng cho một bệnh phụ, có thể bỏ qua phần đó.
- Nếu NGỮ CẢNH cho phép, nên chia thành: Tổng quan bệnh / Nguyên nhân & điều kiện phát sinh / Triệu chứng / Hậu quả / Hướng xử lý & phòng ngừa.
- Chỉ tạo mục khi NGỮ CẢNH có dữ liệu; nếu không có thì bỏ qua mục đó.
""".strip()
    elif answer_mode == "product":
        mode_requirements = """
- Trình bày chi tiết, không trả lời quá ngắn gọn.
- Cố gắn trình bày đầy đủ thông tin sản phẩm, lưu ý, công thức.
- Làm rõ đặc tính sản phẩm, cơ chế (nếu NGỮ CẢNH có), phạm vi tác động.
- CHỈ sử dụng dữ liệu trong NGỮ CẢNH, không được tự bịa thêm liều lượng, cách pha, thời gian cách ly.
""".strip()
    elif answer_mode == "procedure":
        mode_requirements = """
- Diễn giải chi tiết từng bước, mô tả rõ mục đích của mỗi bước nếu NGỮ CẢNH có thông tin.
- Tuyệt đối không tự suy diễn quy trình mới ngoài những gì NGỮ CẢNH cung cấp.
""".strip()
    elif answer_mode == "listing":
        mode_requirements = """
- Tổng hợp đầy đủ các mục xuất hiện trong NGỮ CẢNH, không bỏ sót.
- Không bịa thêm thông tin ngoài NGỮ CẢNH.
""".strip()
    else:
        mode_requirements = """
- Trình bày tự nhiên nhưng chi tiết, không trả lời quá ngắn.
- Nhóm theo chủ đề khi có nhiều ý trong NGỮ CẢNH.
- Không bịa số liệu/liều lượng nếu NGỮ CẢNH không có.
""".strip()

    # RAG STRICT vs SOFT
    if rag_mode == "SOFT":
        system_prompt = (
            "Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN. "
            "Ưu tiên dùng NGỮ CẢNH, nhưng được phép bổ sung kiến thức nông nghiệp phổ biến từ INTERNET khi NGỮ CẢNH thiếu hoặc phân tán để trả lời mạch lạc. "
            "Tuyệt đối không bịa liều lượng, cách pha, thời gian cách ly nếu NGỮ CẢNH không nêu. "
            "Nếu thiếu dữ liệu, hãy trả lời theo hướng khuyến nghị chung và nêu điều kiện/cần xác nhận."
            + BASE_REASONING_PROMPT
        )
    else:
        system_prompt = (
            "Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN. "
            "Luôn ưu tiên sử dụng NGỮ CẢNH được cung cấp nhưng được phép bổ sung kiến thức nông nghiệp phổ biến từ INTERNET khi NGỮ CẢNH thiếu hoặc phân tán để trả lời mạch lạc. "
            "Không bịa số liệu, liều lượng, khuyến cáo chi tiết nếu NGỮ CẢNH không nêu."
            + BASE_REASONING_PROMPT
        )

    user_prompt = f"""
NGỮ CẢNH:
\"\"\"{context}\"\"\"


CÂU HỎI:
\"\"\"{user_query}\"\"\"


HƯỚNG DẪN TRẢ LỜI THEO KIỂU CÂU HỎI ({answer_mode}):
{mode_requirements}

NGUYÊN TẮC CHUNG:
- Ưu tiên thông tin trong NGỮ CẢNH, có thể gom/tổng hợp cho dễ hiểu.
- Không cần viết câu "Tài liệu không đề cập" trừ khi thật sự cần nhấn mạnh. Đặc biệt nếu câu hỏi có liên quan đến vấn đề thủy sinh như cá, tôm, vật nuôi, ... thì phải dứt khoát nhấn mạnh "Tài liệu không đề cập".
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