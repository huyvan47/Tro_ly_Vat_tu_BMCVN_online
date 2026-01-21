import json
from typing import Tuple, Optional

def build_global_enrich_prompt(*, chemical: str, user_query: str) -> Tuple[str, str]:
    system = (
        "Bạn là chuyên gia BVTV/nông học.\n"
        "NHIỆM VỤ: cung cấp KIẾN THỨC NỀN (ngoài tài liệu nội bộ) về HOẠT CHẤT để giúp tư vấn an toàn.\n"
        "RÀNG BUỘC:\n"
        "- Chỉ nói về hoạt chất được nêu.\n"
        "- Không bịa nhãn/PHI/liều/đăng ký. Không khẳng định thay nhà sản xuất.\n"
        "- Trình bày có điều kiện (phụ thuộc cây trồng, giai đoạn, đất, thời tiết, liều, cách dùng).\n"
        "- Ưu tiên nội dung liên quan trực tiếp câu hỏi (ảnh hưởng cây giống/cây con/an toàn/rủi ro).\n"
        "OUTPUT: tiếng Việt, gạch đầu dòng rõ ràng.\n"
        "CẤU TRÚC BẮT BUỘC:\n"
        "1) Tóm tắt đặc tính (nhóm tác động, cách tác động ở mức khái quát)\n"
        "2) Rủi ro/ảnh hưởng tiềm ẩn theo bối cảnh câu hỏi\n"
        "3) Điều kiện làm tăng rủi ro\n"
        "4) Khuyến nghị thực hành an toàn (không nêu liều)\n"
        "5) 2 câu hỏi cần xác minh thêm (nếu muốn kết luận chắc)\n"
    )

    user = {
        "chemical": chemical,
        "user_query": user_query,
    }
    return system, json.dumps(user, ensure_ascii=False)

def query_global_knowledge(
    *,
    client,
    chemical: str,
    user_query: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 900,
) -> str:
    system, payload = build_global_enrich_prompt(chemical=chemical, user_query=user_query)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
