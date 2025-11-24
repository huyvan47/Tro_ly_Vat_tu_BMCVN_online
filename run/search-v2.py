import numpy as np
import re
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

data = np.load("tong-hop-data-phong-vat-tu-fix-24-11.npz", allow_pickle=True)
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




def extract_img_keys(text: str):
    """
    Tìm tất cả (IMG_KEY: xxx) trong text và trả về list key ['xxx', ...]
    """
    return re.findall(r'\(IMG_KEY:\s*([^)]+)\)', text)

def remove_img_keys(text: str):
    """
    Xoá sạch phần '(IMG_KEY: ...)' khỏi text để context gửi vào LLM được sạch sẽ.
    """
    return re.sub(r'-?\s*\(IMG_KEY:[^)]+\)\s*', '', text).strip()

def extract_codes_from_query(text: str):
    """
    Tìm các cụm giống mã vật tư: chữ + số + gạch, hoặc số + gạch (vd: 'cha1000-20', '1000-20', 'cha240-asmil')
    Bạn có thể chỉnh regex cho phù hợp dữ liệu thực tế.
    """
    # Ví dụ đơn giản: từ chứa cả số và dấu gạch ngang
    return re.findall(r'\b[\w]*\d[\w-]*-\d[\w-]*\b', text)

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
- Chỉ sửa lỗi chính tả + chuẩn hoá văn bản hỏi.
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
        "Bạn là Trợ lý Vật tư BMCVN. "
        "Bạn trả lời dựa trên tài liệu nội bộ được cung cấp (NGỮ CẢNH). "
        "Giọng chuyên nghiệp, rõ ràng, có bullet khi cần. "
        "TUYỆT ĐỐI không bịa thông tin ngoài những gì có trong NGỮ CẢNH. "
        "Nếu dữ liệu không đủ để trả lời chính xác, hãy nói rõ 'Không đủ dữ liệu để trả lời chính xác' "
        "và gợi ý người dùng cung cấp thêm thông tin."
    )

    user_prompt = f"""
NGỮ CẢNH (nhiều đoạn tài liệu nội bộ, có thể không đầy đủ):
\"\"\"{context}\"\"\"

CÂU HỎI CỦA NGƯỜI DÙNG:
\"\"\"{user_query}\"\"\"

YÊU CẦU TRẢ LỜI:
- Dùng TỐI ĐA thông tin trong NGỮ CẢNH để trả lời, có thể kết hợp nhiều đoạn khác nhau.
- Không được đưa thông tin không xuất hiện trong NGỮ CẢNH.
- Có thể suy luận, so sánh, tổng hợp từ nhiều đoạn, nhưng không được tự bịa số liệu/quy định mới.
- Trình bày ngắn gọn, rõ ràng, ưu tiên bullet cho các bước/thủ tục.
- Nếu NGỮ CẢNH không đủ, hãy nói rõ 'Không đủ dữ liệu để trả lời chính xác' và giải thích thiếu gì.

DANH SÁCH CÂU HỎI GỢI Ý (CHO PHẦN GỢI Ý CUỐI CÙNG, NẾU CẦN):
{suggestions_text}

SAU KHI TRẢ LỜI:
- (Tuỳ chọn) Có thể gợi ý 1–3 câu hỏi tiếp theo dựa trên danh sách trên, diễn đạt lại cho tự nhiên hơn.
- Các câu gợi ý nên đặt trong ngoặc kép để user dễ copy, ví dụ: "Anh/chị có thể hỏi thêm về quy trình đặt vật tư dự phòng cha240-asmil?".
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # hoặc FT_MODEL nếu bạn muốn dùng model fine-tune
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
    # Có thể normalize để bắt mã tốt hơn
    norm_query = normalize_query(user_query)
    hits = search(user_query, top_k=20)

    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
        }

    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    if not filtered_for_main:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits
        )

        text = (
            "🔍 Không có tài liệu nào đủ độ tương đồng để trả lời chính xác câu hỏi này.\n"
            "Tuy nhiên, bạn có thể tham khảo các chủ đề gần giống dưới đây:\n\n"
            f"{suggestions_text}\n\n"
            "👉 Bạn có thể chọn 1 câu bên trên hoặc hỏi rõ hơn để mình tìm đúng tài liệu."
        )
        return {
            "text": text,
            "img_keys": [],
        }

    # ==============================
    #   XÁC ĐỊNH DOC CHÍNH (PRIMARY)
    # ==============================

    code_candidates = extract_codes_from_query(norm_query.lower())
    primary_doc = None

    if code_candidates:
        # ưu tiên code đầu tiên
        target_code = code_candidates[0]
        for h in filtered_for_main:
            if (target_code in h["question"].lower()) or (target_code in h["answer"].lower()):
                primary_doc = h
                break

    # Nếu không tìm được bằng mã → dùng doc có score cao nhất
    if primary_doc is None:
        primary_doc = filtered_for_main[0]

    # Đảm bảo primary_doc đứng đầu danh sách context
    MAX_CONTEXT_DOCS = 5
    main_context_hits = []

    # thêm primary_doc trước
    main_context_hits.append(primary_doc)

    # thêm các doc khác (không trùng) cho đủ MAX_CONTEXT_DOCS
    for h in filtered_for_main:
        if h is primary_doc:
            continue
        if len(main_context_hits) >= MAX_CONTEXT_DOCS:
            break
        main_context_hits.append(h)

    # ==============================
    #   TẠO CONTEXT + IMG_KEY
    # ==============================

    context_blocks = []

    # 1) IMG_KEY chỉ lấy từ primary_doc
    raw_answer_primary = primary_doc["answer"]
    img_keys_primary = extract_img_keys(raw_answer_primary)

    # 2) Context: từ nhiều DOC, nhưng đã clean IMG_KEY
    for i, h in enumerate(main_context_hits, start=1):
        raw_answer = h["answer"]
        cleaned_answer = remove_img_keys(raw_answer)

        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI MẪU: {h['question']}\n"
            f"CÂU TRẢ LỜI / TÀI LIỆU LIÊN QUAN:\n{cleaned_answer}"
        )
        context_blocks.append(block)

    context = "\n\n------------------------------\n\n".join(context_blocks)

    # ==============================
    #   GỢI Ý CÂU HỎI
    # ==============================

    used_questions = {h["question"] for h in main_context_hits}

    filtered_for_suggest = [
        h for h in hits
        if (h["question"] not in used_questions) and (h["score"] >= MIN_SCORE_SUGGEST)
    ]

    suggestions = filtered_for_suggest[:MAX_SUGGEST]

    if suggestions:
        suggestions_text = "\n".join(
            f"- {h['question']}"
            for h in suggestions
        )
    else:
        suggestions_text = "- (Không có câu gợi ý phù hợp)"

    # ==============================
    #   GỌI LLM (RAG)
    # ==============================

    final_answer = call_finetune_with_context(
        user_query=user_query,
        context=context,
        suggestions_text=suggestions_text
    )

    # Nếu primary_doc không có IMG_KEY nào → tùy bạn:
    #  - hoặc trả list rỗng,
    #  - hoặc fallback: extract từ tất cả main_context_hits.
    if img_keys_primary:
        unique_img_keys = sorted(set(img_keys_primary))
    else:
        # Fallback option (nếu muốn)
        collected = []
        for h in main_context_hits:
            collected.extend(extract_img_keys(h["answer"]))
        unique_img_keys = sorted(set(collected))

    return {
        "text": final_answer,
        "img_keys": unique_img_keys,
    }




# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    q = 'cho tôi biết quy trình thiết kế nhãn riêng'
    res = answer_with_suggestions(q)

    print("\n===== KẾT QUẢ CUỐI CÙNG =====\n")
    print(res["text"])

    print("\nIMG_KEY dùng để truy xuất hình:")
    for k in res["img_keys"]:
        print("-", k)


