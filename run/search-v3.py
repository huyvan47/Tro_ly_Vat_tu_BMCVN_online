import numpy as np
from openai import OpenAI

# ==============================
#       CONFIG
# ==============================

FT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::CdBoxNIT"   # đổi thành model của bạn

MIN_SCORE_MAIN = 0.60       # Tối thiểu để dùng làm context
MIN_SCORE_SUGGEST = 0.50    # Tối thiểu để dùng làm gợi ý
MAX_SUGGEST = 5             # Gợi ý tối đa 5 câu

client = OpenAI(api_key="...")

# ==============================
#       LOAD DATA
# ==============================

data = np.load("tong-hop-data-phong-vat-tu.npz", allow_pickle=True)
EMBS = data["embeddings"]      # (N, d)
QUESTIONS = data["questions"]  # (N,)
ANSWERS = data["answers"]      # (N,)

# ==============================
#       EMBEDDING FUNCTION
# ==============================

def embed_query(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

# ==============================
#     NORMALIZE QUERY
# ==============================

def normalize_query(q: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
Bạn là Query Normalizer cho trợ lý Vật tư BMC.

Nhiệm vụ:
1) Sửa lỗi chính tả, chuẩn hoá câu, bỏ bớt rác ngôn ngữ (chữ kéo dài, icon, từ thừa...) nhưng không sửa các cụm từ giống như mã ví dụ sau đây là các mã: cha240-asmil, cha1000-02logo, bộ pet vàng, ...
2) Bỏ qua các phần không quan trọng như:
   - đại từ nhân xưng (em, anh, chị, mình, tụi em, bên em...)
   - lời chào (hello, xin chào, chào anh/chị...)
   - lời cảm ơn (cảm ơn anh, cảm ơn ạ, thanks...)
   - từ cảm thán/đệm (ơi, với ạ, nha, nhé, ạ...)
   - cụm như: "cho em hỏi", "em muốn hỏi", "anh tư vấn giúp", ...
3) Cố gắng giữ lại đúng "ý nghiệp vụ" mà người dùng đang hỏi (nếu có).

Đồng thời, hãy PHÂN LOẠI loại câu theo 3 nhóm:
- "QUESTION": Người dùng đang hỏi về nghiệp vụ, quy trình, vật tư, kế hoạch, tồn kho, nhà cung cấp, ... (có thể tra trong tài liệu).
- "SMALL_TALK": Người dùng chỉ chào hỏi, cảm ơn, khen/chê, xã giao, nói chuyện vu vơ, KHÔNG có ý định hỏi nghiệp vụ.
- "OTHER": Câu không rõ nghĩa, spam, hoặc không thuộc 2 nhóm trên.

KẾT QUẢ TRẢ VỀ:
- Luôn trả về đúng 1 chuỗi JSON với 2 field:
  {
    "normalized_query": "<chuỗi sau khi chuẩn hoá, nếu không phải QUESTION thì để rỗng>",
    "intent": "<QUESTION|SMALL_TALK|OTHER>"
  }

YÊU CẦU:
- Không giải thích, không thêm chữ ngoài JSON.
"""
            },
            {"role": "user", "content": q}
        ],
    )

    import json

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        # Đảm bảo có đủ field
        normalized = data.get("normalized_query", "").strip()
        intent = data.get("intent", "").strip()
        return {
            "normalized_query": normalized,
            "intent": intent
        }
    except Exception:
        # Fallback: coi như câu hỏi bình thường
        return {
            "normalized_query": q.strip(),
            "intent": "QUESTION"
        }


# ==============================
#        SEARCH ENGINE
# ==============================

def search(query: str, top_k: int = 10):
    # Ở đây query ĐÃ là câu đã chuẩn hoá (string)
    print("search() received query:", query)

    vq = embed_query(query)         # dùng thẳng query string
    sims = EMBS @ vq                # cosine similarity như cũ

    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        results.append({
            "question": str(QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        })

    return results

# ==============================
#   CALL FINE-TUNE FOR ANSWER
# ==============================

def call_finetune_with_context(user_query: str, context: str, suggestions_text: str):
    print("user_query: ", user_query)
    system_prompt = (
        "Bạn là Trợ lý Vật tư BMCVN. Giọng chuyên nghiệp, rõ ràng, "
        "có bullet khi cần. Không bịa thông tin."
    )

    user_prompt = f"""
NGỮ CẢNH (tài liệu nội bộ, có thể không đầy đủ):
\"\"\"{context}\"\"\"

CÂU HỎI CỦA NGƯỜI DÙNG:
\"\"\"{user_query}\"\"\"

YÊU CẦU:
- Giữ nguyên 100% nội dung, không được tóm tắt, không được rút gọn, không được thay đổi văn phong. Trả lời lại y hệt như tôi gửi.
- Ưu tiên dựa trên NGỮ CẢNH.


Gợi ý câu hỏi tiếp theo một cách mềm mại, uyển chuyển, thân thiện và bỏ trong ngoặc kép các câu gợi ý để người dùng dễ ràng hiểu và hỏi đúng câu:
{suggestions_text}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()

# ==============================
#   MAIN PIPELINE (FULL FLOW)
# ==============================

def answer_with_suggestions(user_query: str):
    # B1: Chuẩn hoá + phân loại intent bằng GPT
    norm_result = normalize_query(user_query)
    norm_query = norm_result["normalized_query"]
    intent = norm_result["intent"]

    print("intent:", intent)
    print("norm_query:", norm_query)

    # B2: Nếu là SMALL_TALK -> trả lời nhẹ nhàng, KHÔNG search, KHÔNG top_k
    if intent == "SMALL_TALK":
        # Bạn có thể random 2-3 câu khác nhau cho tự nhiên
        return "Vâng, em cảm ơn anh/chị. Nếu cần hỗ trợ thêm gì về vật tư, anh/chị cứ nhắn cho em nhé ạ. 😊"

    # B3: Nếu là OTHER (lạ, không rõ) -> xin người dùng nói rõ hơn, KHÔNG search
    if intent == "OTHER" or not norm_query:
        return (
            "Hiện tại em chưa hiểu rõ anh/chị đang muốn hỏi về nội dung nào trong phần vật tư.\n"
            "Anh/chị có thể mô tả cụ thể hơn (ví dụ: mã chai, thùng, quy trình, tồn kho, kế hoạch đặt hàng...) để em tra cứu chính xác hơn được không ạ?"
        )

    # B4: Chỉ khi intent == QUESTION mới chạy search + top_k
    hits = search(norm_query, top_k=10)

    if not hits:
        return (
            "Em chưa tìm thấy tài liệu phù hợp với câu hỏi này trong kho dữ liệu hiện tại.\n"
            "Anh/chị mô tả lại cụ thể hơn giúp em (ví dụ: loại vật tư, công đoạn, hoặc câu hỏi chi tiết hơn) để em hỗ trợ tốt hơn nhé."
        )

    # --- Lọc context chính theo threshold ---
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    # Không có doc đủ điểm -> gợi ý top_k (chỉ vì đây là QUESTION)
    if not filtered_for_main:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits
        )
        return (
            "Em chưa tìm được tài liệu nào thật sự khớp 100% với câu hỏi.\n"
            "Tuy nhiên có một số nội dung gần với ý anh/chị, anh/chị tham khảo thử xem có đúng cái mình cần không ạ:\n\n"
            f"{suggestions_text}\n\n"
            "👉 Nếu chưa đúng, anh/chị mô tả rõ hơn (thêm mã vật tư, tên chai/thùng, loại quy trình...) để em tìm lại cho chính xác hơn nhé."
        )

    # Còn lại: có context chính -> xử lý như cũ
    main_hit = filtered_for_main[0]
    context = main_hit["answer"]

    filtered_for_suggest = [
        h for h in hits
        if h["question"] != main_hit["question"] and h["score"] >= MIN_SCORE_SUGGEST
    ]

    suggestions = filtered_for_suggest[:MAX_SUGGEST]

    if suggestions:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in suggestions
        )
    else:
        suggestions_text = "- (Không có câu gợi ý phù hợp)"

    final_answer = call_finetune_with_context(
        user_query=user_query,
        context=context,
        suggestions_text=suggestions_text
    )

    return final_answer


# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    q = "hay quá"
    ans = answer_with_suggestions(q)
    print("\n===== KẾT QUẢ CUỐI CÙNG =====\n")
    print(ans)
