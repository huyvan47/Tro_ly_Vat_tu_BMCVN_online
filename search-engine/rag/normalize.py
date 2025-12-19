def normalize_query(client, q: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                Bạn là Query Normalizer.
                Nhiệm vụ: CHỈ chuẩn hoá câu hỏi người dùng.

                Quy tắc bắt buộc:
                - KHÔNG trả lời câu hỏi.
                - KHÔNG bổ sung thông tin mới (không thêm tên sản phẩm, liều lượng, thời điểm, khuyến cáo...).
                - KHÔNG diễn giải dài dòng, KHÔNG thêm ví dụ.
                - GIỮ NGUYÊN mọi mã sản phẩm/hoạt chất (ví dụ: Maruka, Hariwon, ...).
                - Chỉ sửa lỗi chính tả, viết hoa/thường hợp lý, dấu câu, khoảng trắng.
                - Đầu ra chỉ gồm DUY NHẤT câu hỏi đã chuẩn hoá (không kèm lời giải thích).
                """.strip()
            },
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()