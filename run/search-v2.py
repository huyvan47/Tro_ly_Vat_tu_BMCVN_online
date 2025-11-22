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

data = np.load("tong-hop-data-phong-vat-tu-fix-22-11.npz", allow_pickle=True)
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
Bạn là Query Normalizer.
Nhiệm vụ:
- Sửa lỗi chính tả KHÔNG ĐƯỢC thay đổi bất kỳ ký tự nào bên trong các chuỗi có chứa số và dấu gạch ngang chạy liền nhau (ví dụ: "cha240-asmil", "cha1000-02logo", "450-02", "cha240-04").
- Khi thấy một cụm giống mã (bao gồm chữ cái + số + dấu gạch ngang, hoặc số + gạch ngang), phải giữ NGUYÊN Y HỆT, không sửa chính tả, không thêm/bớt khoảng trắng.
- Dựa tối đa vào NGỮ CẢNH để trả lời.
- Không bịa thông tin mới.
- Có thể diễn giải lại cho dễ hiểu, nhưng không được thêm nội dung không có trong ngữ cảnh.
                """
            },
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()

# ==============================
#        SEARCH ENGINE
# ==============================

def search(query: str, top_k: int = 10):
    norm_query = normalize_query(query)
    print("norm_query:", norm_query)

    vq = embed_query(norm_query)
    sims = EMBS @ vq         # cosine similarity

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
    print("context: ", context)
    # print("suggestions_text: ", suggestions_text)
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
    # Lấy nhiều hơn 5 để lọc bớt
    hits = search(user_query, top_k=10)

    if not hits:
        return "Không tìm thấy dữ liệu phù hợp."

    # --- Lọc context chính theo threshold ---
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    if not filtered_for_main:

        # Tạo danh sách gợi ý từ toàn bộ top_k
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits
        )

        return (
            "🔍 Không có tài liệu nào đủ độ tương đồng để trả lời chính xác câu hỏi này.\n"
            "Tuy nhiên, bạn có thể tham khảo các chủ đề gần giống dưới đây:\n\n"
            f"{suggestions_text}\n\n"
            "👉 Bạn có thể chọn 1 câu bên trên hoặc hỏi rõ hơn để mình tìm đúng tài liệu."
        )

    # Câu khớp nhất
    main_hit = filtered_for_main[0]
    context = main_hit["answer"]

    answer_text = context
    # --- Lọc câu gợi ý ---
    filtered_for_suggest = [
        h for h in hits
        if h["question"] != main_hit["question"] and h["score"] >= MIN_SCORE_SUGGEST
    ]

    suggestions = filtered_for_suggest[:MAX_SUGGEST]

    # Format phần gợi ý
    if suggestions:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in suggestions
        )
    else:
        suggestions_text = "- (Không có câu gợi ý phù hợp)"

    # Gọi fine-tune
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
    q = "đặt vật tư dự phòng"
    ans = answer_with_suggestions(q)
    print("\n===== KẾT QUẢ CUỐI CÙNG =====\n")
    print(ans)
