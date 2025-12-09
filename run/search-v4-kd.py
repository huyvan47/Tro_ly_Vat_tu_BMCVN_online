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
TOP_K_RERANK = 40          # Số doc tối đa đưa vào LLM để rerank

client = OpenAI(api_key="...")

# ==============================
#       LOAD DATA
# ==============================

data = np.load("data-kinh-doanh-nam-benh.npz", allow_pickle=True)

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
    # ví dụ: cha240-06, 450-02, cha240-asmil-01...
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

def route_query(user_query: str) -> str:
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
"""
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


# ==============================
#   LLM RERANK
# ==============================

def llm_rerank(norm_query: str, results: list, top_k_rerank: int = TOP_K_RERANK):
    """
    Rerank top_k_rerank documents bằng LLM mini.
    Sử dụng thêm trường include_in_context:
    - Những doc có include_in_context == true sẽ được ưu tiên cho CONTEXT.
    - Vẫn giữ lại các doc khác ở phía sau cho việc gợi ý / fallback.
    LUÔN trả về 1 list results hợp lệ (không crash).
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
    print("docs_block: ", docs_block)

    # ===============================
    # 2) Build prompts
    # ===============================
    system_prompt = (
        "Bạn là LLM dùng để RERANK tài liệu cho hệ thống hỏi đáp bệnh cây / nông nghiệp BMCVN. "
        "Bạn CHỈ được trả về JSON THUẦN, không có code block hay markdown."
    )

    user_prompt = f"""
CÂU HỎI:
\"\"\"{norm_query}\"\"\"


CÁC TÀI LIỆU ỨNG VIÊN:

{docs_block}

YÊU CẦU:
- Với mỗi DOC, hãy:
  (1) Chấm điểm liên quan từ 0–1
  (2) Quyết định có nên đưa DOC này vào NGỮ CẢNH trả lời hay không bằng true/false
- Ưu tiên đưa vào NGỮ CẢNH các DOC thuộc các nhóm:
  - Cách trị / thuốc
  - Triệu chứng nhận biết
  - Nguyên nhân & lây lan
  - Biện pháp canh tác / cảnh báo

- Trả về DUY NHẤT JSON, ví dụ:
[
  {{ "doc_index": 0, "score": 0.92, "include_in_context": true }},
  {{ "doc_index": 1, "score": 0.30, "include_in_context": false }}
]
- KHÔNG dùng ```json hay markdown.
- KHÔNG giải thích.
- Nếu không thể trả JSON đúng, trả [].
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
    # 5) Parse JSON
    # ===============================
    try:
        ranking = json.loads(cleaned)
        if not isinstance(ranking, list):
            raise ValueError("JSON is not a list")
    except Exception:
        print("LLM rerank JSON parse failed → fallback to original order.")
        return results

    # ===============================
    # 6) Gán rerank_score + include_in_context
    # ===============================
    idx_to_result = {i: candidates[i] for i in range(len(candidates))}

    # set default cho tất cả candidates
    for h in candidates:
        h["rerank_score"] = 0.0
        h["include_in_context"] = False

    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if di not in idx_to_result:
                continue
            score = float(item.get("score", 0))
            include_flag = bool(item.get("include_in_context", False))

            idx_to_result[di]["rerank_score"] = score
            idx_to_result[di]["include_in_context"] = include_flag
        except Exception as e:
            print("Error processing ranking item:", e)
            continue

    # ===============================
    # 7) Xếp lại kết quả theo thứ tự LLM
    # ===============================
    used = set()
    reranked = []

    try:
        for item in ranking:
            di = int(item.get("doc_index", -1))
            if di in idx_to_result and di not in used:
                reranked.append(idx_to_result[di])
                used.add(di)
    except Exception as e:
        print("LLM ranking processing error:", e)
        return results

    # Thêm các doc chưa có trong ranking (nếu có)
    for i, h in enumerate(candidates):
        if i not in used:
            reranked.append(h)

    # Nếu còn doc ngoài top_k_rerank thì nối vào đuôi
    if len(results) > len(candidates):
        reranked.extend(results[len(candidates):])

    # ===============================
    # 8) Ưu tiên doc include_in_context lên đầu
    # ===============================
    selected = [h for h in reranked if h.get("include_in_context") is True]
    others = [h for h in reranked if not h.get("include_in_context")]

    if selected:
        final_results = selected + others
    else:
        final_results = reranked

    return final_results


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
        """
        Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN.

        NGUYÊN TẮC SỬ DỤNG THÔNG TIN:
        - ƯU TIÊN cao nhất: NGỮ CẢNH (các DOC được cung cấp). Nếu NGỮ CẢNH có thông tin, phải dùng đúng theo đó.
        - ĐƯỢC PHÉP dùng thêm kiến thức nền nông nghiệp / bảo vệ thực vật phổ biến, nếu:
        • Không mâu thuẫn với NGỮ CẢNH.
        • Mang tính khái quát, không đi vào chi tiết nhạy cảm (liều phun, thời gian cách ly, tên nhãn thương mại ngoài NGỮ CẢNH).
        • Phù hợp với câu hỏi và logic chuyên môn (ví dụ: phân loại hoạt chất theo nhóm FRAC, cơ chế tác động chung, nguyên tắc canh tác cơ bản).

        - KHÔNG ĐƯỢC:
        • Bịa thêm số liệu cụ thể (liều lượng, nồng độ, số lần phun, thời gian cách ly…) khi NGỮ CẢNH không có.
        • Suy diễn ra khuyến cáo quá chi tiết hoặc trái với nhãn thuốc thông thường.
        • Nói điều gì đi ngược hoặc phủ nhận thông tin trong NGỮ CẢNH.

        Quy trình trả lời (bạn làm thầm, KHÔNG in ra):

        BƯỚC 1 – Nhận diện LOẠI THÔNG TIN có trong NGỮ CẢNH (có thể có nhiều loại cùng lúc):
        - Thông tin bệnh hại (tác nhân, triệu chứng, lây lan, phòng trị)
        - Thông tin sản phẩm / thuốc / phân
        - Thông tin quy trình / hướng dẫn sử dụng
        - Thông tin cảnh báo / an toàn
        - Thông tin lý thuyết / cơ chế / khái niệm
        - Thông tin canh tác thực hành

        BƯỚC 2 – Với mỗi LOẠI THÔNG TIN phát hiện được:
        - Nếu có đủ dữ liệu → trình bày thành 1 mục riêng, có tiêu đề rõ ràng.
        - Nếu không có → KHÔNG tự suy diễn.

        BƯỚC 3 – Cấu trúc câu trả lời:
        - Không dùng một sườn cố định cứng nhắc.
        - Mỗi mục chỉ xuất hiện nếu NGỮ CẢNH thật sự có dữ liệu cho mục đó.
        - Ưu tiên trình bày theo nhóm:
        • Tổng quan
        • Bệnh hại & triệu chứng (nếu có)
        • Thuốc / hoạt chất / nhóm FRAC (nếu có)
        • Cách sử dụng / phác đồ (nếu có)
        • Canh tác & quản lý (nếu có)
        • Cảnh báo & an toàn (nếu có)

        YÊU CẦU BẮT BUỘC:
        - Không được trả lời “Tài liệu không đề cập” nếu trong bất kỳ DOC nào có thông tin liên quan.
        - Chỉ được viết những mục mà NGỮ CẢNH có dữ liệu.
        - Không suy diễn ngoài dữ liệu.
        - Viết ưu tiên cho nông dân & kỹ thuật hiện trường: rõ ràng – hành động được.
        """
    )

    user_prompt = f"""
NGỮ CẢNH:
\"\"\"{context}\"\"\"


CÂU HỎI:
\"\"\"{user_query}\"\"\"


YÊU CẦU:
- Trả lời theo đúng định hướng trong SYSTEM PROMPT (tập trung vào phác đồ xử lý thực tế, SOP ngoài vườn).
- Chỉ sử dụng thông tin có trong NGỮ CẢNH. KHÔNG tự bịa thêm.
- Tuyệt đối KHÔNG được trả lời "Tài liệu không đề cập" nếu trong NGỮ CẢNH có bất kỳ mô tả triệu chứng, dấu hiệu bệnh, hay biểu hiện nào dù chỉ ở 1 DOC phụ.
- Nếu NGỮ CẢNH không cung cấp thông tin cho một mục nào đó (ví dụ: nhóm FRAC, thời điểm phun, cảnh báo nhà kính...), hãy ghi rõ "Tài liệu không đề cập phần này" thay vì tự suy diễn.
- Ưu tiên trình bày theo các mục: (1) Tóm tắt, (2) Nguyên nhân & lây lan, (3) Triệu chứng, (4) Nguyên tắc xử lý, (5) Phác đồ thuốc, (6) Biện pháp canh tác, (7) Cảnh báo, (8) Checklist hành động — nhưng chỉ điền những mục mà NGỮ CẢNH có dữ liệu.
- Cuối câu trả lời, gợi ý thêm 1–3 câu hỏi liên quan, ưu tiên lấy ý từ danh sách:
{suggestions_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content.strip()


# ==============================
#   CHỌN SỐ DOC CHO CONTEXT
# ==============================

def choose_adaptive_max_ctx(hits_reranked, is_listing: bool = False):
    """
    Quyết định số lượng DOC đưa vào context trả lời dựa trên LLM rerank score (0–1).
    Nếu is_listing=True → cho phép trả về nhiều DOC hơn (tối đa 20).
    """
    print('is_listing: ', is_listing)
    scores = [h.get("rerank_score", 0) for h in hits_reranked[:4]]
    scores += [0] * (4 - len(scores))

    s1, s2, s3, s4 = scores

    # Nếu là câu hỏi dạng LIỆT KÊ → scale lên nhiều hơn
    if is_listing:
        # Liên quan mạnh → cho đọc tối đa 20 DOC
        if s1 >= 0.75 and s2 >= 0.65 and s3 >= 0.55:
            return 20
        # Liên quan vừa → 15 DOC
        if s1 >= 0.65 and s2 >= 0.55:
            return 15
        # Liên quan hơi yếu → 10 DOC
        return 10

    # Ngược lại: câu hỏi thường → dùng ngưỡng cũ, context nhỏ để tránh nhiễu
    # Liên quan cực mạnh → cho LLM đọc nhiều doc
    if s1 >= 0.90 and s2 >= 0.80 and s3 >= 0.75 and s4 >= 0.70:
        return 10
    # Liên quan mạnh
    if s1 >= 0.85 and s2 >= 0.75 and s3 >= 0.70:
        return 7
    # Liên quan vừa
    if s1 >= 0.80 and s2 >= 0.65:
        return 5

    # Yếu → chỉ 5 doc (an toàn)
    return 5


# ==============================
#   MAIN PIPELINE
# ==============================

def is_listing_query(q):
    t = q.lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê",
        "kể tên", "bao nhiêu", "tổng", "có bao nhiêu", "gồm"
    ])


def answer_with_suggestions(user_query: str):
    # 0. Router LLM + rule
    route = route_query(user_query)
    if route == "GLOBAL":
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia BVTV, giải thích khái niệm, viết tắt, định nghĩa một cách ngắn gọn, chuẩn sách giáo trình."},
                {"role": "user", "content": user_query},
            ],
        )
        return {"text": resp.choices[0].message.content.strip(), "img_keys": []}
    norm_query = normalize_query(user_query)

    is_list = is_listing_query(norm_query)
    # Nếu là câu hỏi dạng LIỆT KÊ -> cho search rộng hơn
    if is_list:
        hits = search(norm_query, top_k=50)
    else:
        hits = search(norm_query, top_k=20)

    if not hits:
        return {"text": "Không tìm thấy dữ liệu phù hợp.", "img_keys": []}

    # Bước 1: dùng MIN_SCORE_MAIN chỉ để detect “không liên quan”
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    if not filtered_for_main:
        # Không có doc nào đủ gần → trả lời "không đủ dữ liệu"
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits[:10]
        )
        return {
            "text": "Không có tài liệu đủ độ tương đồng.\n\nGợi ý:\n" + suggestions_text,
            "img_keys": []
        }

    # Quyết định số doc tối đa cho context (dựa trên rerank_score)
    max_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)

    # Ưu tiên những doc include_in_context == True
    context_candidates = [h for h in filtered_for_main if h.get("include_in_context", False)]

    # Nếu LLM không chọn doc nào (toàn False) → fallback dùng filtered_for_main
    if not context_candidates:
        context_candidates = filtered_for_main

    # Xác định primary_doc trong context_candidates (ưu tiên code, nếu có)
    code_candidates = extract_codes_from_query(norm_query)
    primary_doc = None

    if code_candidates:
        target = code_candidates[0].lower()
        for h in context_candidates:
            if target in h["question"].lower() or target in h["answer"].lower():
                primary_doc = h
                break

    if primary_doc is None:
        primary_doc = context_candidates[0]

    # Build main_hits từ context_candidates
    main_hits = [primary_doc]
    for h in context_candidates:
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

    # Gợi ý câu hỏi liên quan
    used = {h["question"] for h in main_hits}
    suggest = [
        h for h in hits
        if h["question"] not in used and h["score"] >= MIN_SCORE_SUGGEST
    ][:MAX_SUGGEST]

    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    # Gọi LLM sinh ANSWER
    final_answer = call_finetune_with_context(user_query, context, suggestions_text)

    # IMG_KEY
    img_keys = extract_img_keys(primary_doc["answer"])

    return {"text": final_answer, "img_keys": img_keys}


# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    q = "nói về cơ chế vị độc của thuốc"
    res = answer_with_suggestions(q)

    print("\n===== KẾT QUẢ =====\n")
    print(res["text"])

    print("\nIMG_KEY:")
    print(res["img_keys"])
