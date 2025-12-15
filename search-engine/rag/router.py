def route_query(client, user_query: str) -> str:
    """
    Trả về:
    - "RAG"    : ưu tiên dùng tài liệu nội bộ
    - "GLOBAL" : để model tự trả lời bằng kiến thức nền
    """
    system_prompt = """
Bạn là bộ phân luồng câu hỏi cho hệ thống trợ lý nông nghiệp của BMCVN.

NHIỆM VỤ:
- Nếu câu hỏi chủ yếu là KHÁI NIỆM CHUNG (ví dụ: SC là gì, EC là gì, cơ chế thuốc trừ nấm nói chung, định nghĩa pH...) 
  và có thể trả lời tốt bằng kiến thức nông nghiệp phổ biến trên thế giới → trả lời đúng 1 từ: GLOBAL.
- Nếu câu hỏi liên quan đến:
  • sản phẩm, thuốc, quy trình, tài liệu nội bộ của BMCVN
  • bệnh, cây trồng, phác đồ, SOP có thể có trong tài liệu nội bộ
  → trả lời đúng 1 từ: RAG.

Chỉ trả lời RAG hoặc GLOBAL, không thêm chữ nào khác.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )
    ans = resp.choices[0].message.content.strip().upper()
    return "GLOBAL" if "GLOBAL" in ans else "RAG"