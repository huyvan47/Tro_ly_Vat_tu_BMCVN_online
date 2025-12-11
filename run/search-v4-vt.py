import numpy as np
import re
import json
from openai import OpenAI

# ==============================
#       CONFIG
# ==============================

MIN_SCORE_MAIN = 0.35
MIN_SCORE_SUGGEST = 0.4
MAX_SUGGEST = 3

USE_LLM_RERANK = True      # Bật/tắt rerank bằng LLM
TOP_K_RERANK = 60         # Số doc tối đa đưa vào LLM để rerank

client = OpenAI(api_key="...")

# ==============================
#       LOAD DATA
# ==============================

data = np.load("tong-hop-data-phong-vat-tu-fix-25-11-QAAL.npz", allow_pickle=True)

EMBS = data["embeddings"]
QUESTIONS = data["questions"]
ANSWERS = data["answers"]
ALT_QUESTIONS = data["alt_questions"]
# TEXTS_FOR_EMBEDDING = data["texts_for_embedding"]

CATEGORY = data.get("category", None)
TAGS = data.get("tags", None)

# ==============================
#   HELPER FUNCTIONS
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
    return re.findall(r'\(IMG_KEY:\s*([^)]+)\)', text)


def remove_img_keys(text: str):
    return re.sub(r'-?\s*\(IMG_KEY:[^)]+\)\s*', '', text).strip()


def extract_codes_from_query(text: str):
    return re.findall(r'\b[\w]*\d[\w-]*-\d[\w-]*\b', text)


# ==============================
#   NORMALIZE QUERY (LLM)
# ==============================

def normalize_query(q: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                Bạn là Query Normalizer.
                Không thay đổi các mã như cha240-06, 450-02, cha240-asmil...
                Chỉ sửa lỗi chính tả và chuẩn hoá văn bản.
                """
            },
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()


# ==============================
#   CATEGORY / TAG BOOSTING
# ==============================

def detect_query_category(text: str):
    """Trả về category ưu tiên dựa trên từ khoá xuất hiện trong query."""
    t = text.lower()

    if any(k in t for k in ["misa", "mua hàng", "nhập kho", "hóa đơn"]):
        return "quy_trinh_mua_hang_misa"

    if any(k in t for k in ["kế hoạch", "dự trù", "vật tư", "kế hoạch sử dụng"]):
        return "ke_hoach_vat_tu"

    if any(k in t for k in ["chai", "pet", "hdpe", "nhôm", "nhom"]):
        return "chai"

    if any(k in t for k in ["thùng", "carton", "thung"]):
        return "thung"

    if any(k in t for k in ["tem", "nhãn", "màng", "túi"]):
        return "nhan_mang_tui"

    return None


def boost_similarity_scores(raw_sims, query_text):
    """Tăng điểm cho doc thuộc category hoặc tags phù hợp."""
    sims = raw_sims.copy()
    N = len(sims)
    qcat = detect_query_category(query_text)
    qtext = query_text.lower()

    for i in range(N):

        # 1) BOOST CATEGORY
        if CATEGORY is not None and qcat is not None and CATEGORY[i] == qcat:
            sims[i] += 0.12   # boost mạnh

        # 2) BOOST TAGS
        if TAGS is not None:
            taglist = str(TAGS[i]).lower().split("|")
            if any(t in qtext for t in taglist if t.strip()):
                sims[i] += 0.05   # boost nhẹ

    return sims

def llm_rerank(norm_query: str, results: list, top_k_rerank: int = TOP_K_RERANK):
    """
    Rerank top_k_rerank documents bằng LLM mini.
    LUÔN trả về 1 list results hợp lệ (không bao giờ crash).
    """

    if not results or len(results) == 1:
        return results

    # Giới hạn số doc gửi vào LLM
    candidates = results[:top_k_rerank]

    # ===============================
    # 1) Build docs text
    # ===============================
    doc_texts = []
    for i, h in enumerate(candidates):
        ans = str(h["answer"])
        if len(ans) > 500:
            ans = ans[:500] + " ..."
        block = (
            f"[DOC {i}]\n"
            f"QUESTION: {h['question']}\n"
            f"ALT_QUESTION: {h['alt_question']}\n"
            f"ANSWER_SNIPPET:\n{ans}"
        )
        doc_texts.append(block)

    docs_block = "\n\n------------------------\n\n".join(doc_texts)
    print('docs_block: ', docs_block)

    # ===============================
    # 2) Build prompts
    # ===============================
    system_prompt = (
        "Bạn là LLM dùng để RERANK tài liệu cho hệ thống vật tư BMCVN. "
        "Bạn CHỈ được trả về JSON THUẦN, không có code block hay markdown."
    )

    user_prompt = f"""
CÂU HỎI:
\"\"\"{norm_query}\"\"\"

CÁC TÀI LIỆU ỨNG VIÊN:

{docs_block}

YÊU CẦU:
- Chấm điểm LIÊN QUAN từng DOC trong khoảng 0–1.
- Trả về DUY NHẤT JSON, ví dụ:
[
  {{"doc_index": 0, "score": 0.92}},
  {{"doc_index": 1, "score": 0.85}}
]
- KHÔNG dùng ```json hoặc bất kỳ code block nào.
- KHÔNG giải thích thêm.
- Nếu không thể trả JSON đúng, TRẢ JSON RỖNG: [].
""".strip()

    # ===============================
    # 3) Gọi LLM
    # ===============================
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM call error:", e)
        return results

    # ===============================
    # 4) Làm sạch output
    # ===============================
    cleaned = content.replace("```json", "").replace("```", "").strip()

    print("\n===== CLEANED LLM OUTPUT =====")
    print(cleaned)
    print("==============================\n")

    # ===============================
    # 5) Tìm JSON trong output
    # ===============================
    try:
        ranking = json.loads(cleaned)
        if not isinstance(ranking, list):
            raise ValueError("JSON is not a list")
    except Exception:
        print("LLM rerank JSON parse failed → fallback to original order.")
        return results

    # ===============================
    # 6) Build mapping
    # ===============================
    idx_to_result = {i: candidates[i] for i in range(len(candidates))}
    # Gán rerank_score cho từng doc nếu có
    for item in ranking:
        di = int(item.get("doc_index", -1))
        score = float(item.get("score", 0))
        if di in idx_to_result:
            idx_to_result[di]["rerank_score"] = score

    # Những doc không có trong JSON → rerank_score = 0
    for i in range(len(candidates)):
        if "rerank_score" not in candidates[i]:
            candidates[i]["rerank_score"] = 0.0

    used = set()
    reranked = []

    # ===============================
    # 7) Xếp lại kết quả
    # ===============================
    try:
        for item in ranking:
            di = int(item.get("doc_index", -1))
            if di in idx_to_result and di not in used:
                reranked.append(idx_to_result[di])
                used.add(di)
    except Exception as e:
        print("LLM ranking processing error:", e)
        return results

    # Thêm các doc chưa dùng
    for i, h in enumerate(candidates):
        if i not in used:
            reranked.append(h)

    # Nếu còn doc ngoài top_k_rerank thì nối lại
    if len(results) > len(candidates):
        reranked.extend(results[len(candidates):])

    return reranked



# ==============================
#        SEARCH ENGINE
# ==============================

def search(norm_query: str, top_k=40):
    print("norm_query:", norm_query)

    vq = embed_query(norm_query)
    raw_sims = EMBS @ vq

    # Boost theo category/tags
    sims = boost_similarity_scores(raw_sims, norm_query)

    # Chọn top-k
    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        results.append({
            "question": str(QUESTIONS[i]),
            "alt_question": str(ALT_QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        })

    # ==========================
    #   RERANK BẰNG LLM MINI
    # ==========================
    if USE_LLM_RERANK and len(results) > 1:
        results = llm_rerank(norm_query, results, TOP_K_RERANK)

    return results


# ==============================
#   CALL FINE-TUNE FOR ANSWER
# ==============================

def call_finetune_with_context(user_query, context, suggestions_text):
    system_prompt = (
        "Bạn là Trợ lý Vật tư BMCVN. Trả lời dựa hoàn toàn trên NGỮ CẢNH.\n"
        "- Luôn cố gắng tổng hợp đầy đủ các ý liên quan trong NGỮ CẢNH.\n"
        "- Viết dễ hiểu cho bà con nông dân hoặc nhân viên vật tư.\n"
    )

    user_prompt = f"""
NGỮ CẢNH:
\"\"\"{context}\"\"\"

CÂU HỎI:
\"\"\"{user_query}\"\"\"

YÊU CẦU:
- Trả lời chi tiết, rõ ràng.
- Sử dụng TẤT CẢ các thông tin liên quan trong NGỮ CẢNH, không chỉ dựa trên một DOC.
- Không bịa. Nếu NGỮ CẢNH không nói tới phần nào thì nói rõ là tài liệu không đề cập.
- Không được trả lời sai mã chai, ví dụ: Chai 240-ASMIL, ...
- Ưu tiên dùng bullet cho từng ý chính.
- Gợi ý thêm 1–3 câu hỏi từ danh sách:
{suggestions_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        max_completion_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content.strip()

def choose_adaptive_max_ctx(hits_reranked, is_listing: bool = False): 
    print('is_listing: ', is_listing)

    scores = [h.get("rerank_score", 0) for h in hits_reranked[:4]]
    scores += [0] * (4 - len(scores))
    s1, s2, s3, s4 = scores

    # LISTING MODE (trả về danh sách)
    if is_listing:
        if s1 >= 0.75 and s2 >= 0.65 and s3 >= 0.55:
            return 25
        if s1 >= 0.65 and s2 >= 0.55:
            return 20
        return 15

    # QA MODE (không listing) – tăng mạnh lên 10
    # Query mạnh → 14 DOC
    if s1 >= 0.90 and s2 >= 0.80 and s3 >= 0.75 and s4 >= 0.70:
        return 14

    # Query khá mạnh → 12 DOC
    if s1 >= 0.85 and s2 >= 0.75 and s3 >= 0.70:
        return 12

    # Query trung bình → 10 DOC
    if s1 >= 0.80 and s2 >= 0.65:
        return 10

    # Mặc định (yếu) → cũng lấy 10 DOC (thay vì 6)
    return 10



# ==============================
#   MAIN PIPELINE
# ==============================

def is_listing_query(q):
    t = q.lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê", "kể tên", "bao nhiêu", "tổng", "có bao nhiêu", "gồm"
    ])

def answer_with_suggestions(user_query: str):
    norm_query = normalize_query(user_query)

    is_list = is_listing_query(norm_query)
    # Nếu là câu hỏi dạng LIỆT KÊ -> cho search rộng hơn
    if is_list:
        hits = search(norm_query, top_k=50)
    else:
        hits = search(norm_query, top_k=20)

    # print('hits: ', hits)

    if not hits:
        return {"text": "Không tìm thấy dữ liệu phù hợp.", "img_keys": []}

    # lọc doc đủ score
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    if not filtered_for_main:
        suggestions_text = "\n".join(f"- {h['question']} (score={h['score']:.2f})" for h in hits)
        return {
            "text": "Không có tài liệu đủ độ tương đồng.\n\nGợi ý:\n" + suggestions_text,
            "img_keys": []
        }

    # Xác định doc chính
    code_candidates = extract_codes_from_query(norm_query)
    primary_doc = None

    if code_candidates:
        target = code_candidates[0].lower()
        for h in filtered_for_main:
            if target in h["question"].lower() or target in h["answer"].lower():
                primary_doc = h
                break

    if primary_doc is None:
        primary_doc = filtered_for_main[0]

    # Tạo context
    # Tạo context
    max_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)

    # 2) Build context
    main_hits = [primary_doc]
    for h in filtered_for_main:
        if h is not primary_doc and len(main_hits) < max_ctx:
            main_hits.append(h)

    # Tạo context blocks
    blocks = []
    for i, h in enumerate(main_hits, 1):
        clean_ans = remove_img_keys(h["answer"])
        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI: {h['question']}\n"
            f"HỎI KHÁC: {h['alt_question']}\n"
            f"NỘI DUNG:\n{clean_ans}"
        )
        blocks.append(block)

    context = "\n\n--------------------\n\n".join(blocks)

    # Gợi ý
    used = {h["question"] for h in main_hits}
    suggest = [h for h in hits if h["question"] not in used and h["score"] >= MIN_SCORE_SUGGEST][:MAX_SUGGEST]

    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    # Gọi LLM
    final_answer = call_finetune_with_context(user_query, context, suggestions_text)

    # IMG_KEY
    img_keys = extract_img_keys(primary_doc["answer"])

    return {"text": final_answer, "img_keys": img_keys}


# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    q = "các đầu mục cần kiểm tra trong 1 thiết kế nhãn"
    res = answer_with_suggestions(q)

    print("\n===== KẾT QUẢ =====\n")
    print(res["text"])

    print("\nIMG_KEY:")
    print(res["img_keys"])
