def call_finetune_with_context(client, user_query, context, suggestions_text, answer_mode: str = "general", rag_mode: str = "STRICT"):
    print('answer_mode:', answer_mode)
    # Mode requirements (giữ nguyên tinh thần code v4 của anh)
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
            "Ưu tiên dùng NGỮ CẢNH, nhưng được phép bổ sung kiến thức nông nghiệp phổ biến "
            "khi NGỮ CẢNH thiếu hoặc phân tán để trả lời mạch lạc. "
            "Tuyệt đối không bịa liều lượng, cách pha, thời gian cách ly nếu NGỮ CẢNH không nêu. "
            "Nếu thiếu dữ liệu, hãy trả lời theo hướng khuyến nghị chung và nêu điều kiện/cần xác nhận."
        )
    else:
        system_prompt = (
            "Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN. "
            "Luôn ưu tiên sử dụng NGỮ CẢNH được cung cấp. "
            "Không bịa số liệu, liều lượng, khuyến cáo chi tiết nếu NGỮ CẢNH không nêu."
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
- Không cần viết câu "Tài liệu không đề cập" trừ khi thật sự cần nhấn mạnh.
- Cuối câu trả lời, nếu phù hợp, gợi ý thêm 1–3 câu hỏi liên quan, ưu tiên lấy ý từ danh sách:
{suggestions_text}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_completion_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return resp.choices[0].message.content.strip()