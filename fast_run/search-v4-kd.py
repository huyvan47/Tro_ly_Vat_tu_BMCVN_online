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
TOP_K_RERANK = 60          # Số doc tối đa đưa vào LLM để rerank

client = OpenAI(api_key="...")

# ==============================
#       LOAD DATA
# ==============================

data = np.load("data-kinh-doanh-nam-benh-full-fix.npz", allow_pickle=True)

EMBS = data["embeddings"]
QUESTIONS = data["questions"]
ANSWERS = data["answers"]
ALT_QUESTIONS = data["alt_questions"]

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
                Không thay đổi các mã sản phẩm hoặc hoạt chất như Kenbast 15SL, glufosinate_amonium, ...
                Chỉ sửa lỗi chính tả và chuẩn hoá văn bản.
                """
            },
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()


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

    sims = raw_sims
    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        item = {
            "question": str(QUESTIONS[i]),
            "alt_question": str(ALT_QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        }
        # gắn thêm category, tags nếu có để dùng sau này (nếu muốn)
        if CATEGORY is not None:
            item["category"] = str(CATEGORY[i])
        if TAGS is not None:
            item["tags"] = str(TAGS[i])
        results.append(item)

    # ==========================
    #   RERANK BẰNG LLM MINI
    # ==========================
    if USE_LLM_RERANK and len(results) > 1:
        results = llm_rerank(norm_query, results, TOP_K_RERANK)

    return results


# ==============================
#   MODE TRẢ LỜI LINH HOẠT
# ==============================

def detect_answer_mode(user_query: str, primary_doc: dict, is_listing: bool) -> str:
    """
    Xác định kiểu câu trả lời mong muốn:
    - 'listing'   : liệt kê / tổng hợp
    - 'disease'   : nói về bệnh, triệu chứng, phòng trị
    - 'product'   : nói về thuốc/sản phẩm
    - 'procedure' : nói về quy trình, thao tác
    - 'general'   : kiến thức chung
    """
    if is_listing:
        return "listing"

    text = (user_query + " " + primary_doc.get("question", "")).lower()
    cat = primary_doc.get("category", "").lower()

    # 1) Bệnh / triệu chứng / phòng trị
    if any(kw in text for kw in [
        "bệnh", "triệu chứng", "dấu hiệu", "phòng trị", "phòng trừ",
        "nứt thân", "xì mủ", "thối rễ", "cháy lá", "thán thư", "ghẻ"
    ]) or "benh" in cat:
        return "disease"

    # 2) Thuốc / sản phẩm
    if any(kw in text for kw in [
        "thuốc", "đặc trị", "sc", "wp", "ec", "sl",
        "dùng để làm gì", "tác dụng", "hoạt chất"
    ]) or "san_pham" in cat or "thuoc" in cat:
        return "product"

    # 3) Quy trình / thao tác / hướng dẫn
    if any(kw in text for kw in [
        "quy trình", "quy trinh", "cách làm", "hướng dẫn",
        "các bước", "làm thế nào để", "phương pháp", "thí nghiệm",
        "phân lập", "giám định", "lây bệnh trong phòng thí nghiệm"
    ]) or "quy_trinh" in cat:
        return "procedure"

    return "general"


def call_finetune_with_context(user_query, context, suggestions_text, answer_mode: str = "general"):
    # Hướng dẫn riêng theo mode
    if answer_mode == "disease":
        mode_requirements = """
- Trình bày chi tiết, dễ hiểu cho người trồng. Không trả lời kiểu 1–2 câu là xong.
- Nếu NGỮ CẢNH không mô tả chi tiết triệu chứng cho một bệnh phụ, có thể bỏ qua phần đó, KHÔNG cần viết câu "không có mô tả" trừ khi thật sự cần nhấn mạnh.
- Nếu NGỮ CẢNH cho phép, nên chia thành: Tổng quan bệnh / Nguyên nhân & điều kiện phát sinh / Triệu chứng / Hậu quả / Hướng xử lý & phòng ngừa.
- Mỗi mục nên có mô tả rõ ràng, có thể đưa thêm ví dụ thực tế nếu NGỮ CẢNH có thông tin.
- Chỉ tạo mục khi NGỮ CẢNH có dữ liệu; nếu không có thì bỏ qua mục đó.
"""
    elif answer_mode == "product":
        mode_requirements = """
- Trình bày chi tiết, không trả lời quá ngắn gọn.
- Làm rõ đặc tính sản phẩm, cơ chế (nếu NGỮ CẢNH có), phạm vi tác động, bộ phận cây bị bệnh mà thuốc có hiệu lực.
- Nếu NGỮ CẢNH cho phép, mô tả bối cảnh sử dụng thực tế: thời điểm, điều kiện, loại bệnh liên quan.
- Mỗi ý chính nên có 1–2 câu giải thích để người dùng hiểu vì sao sản phẩm có tác dụng như vậy.
- CHỈ sử dụng dữ liệu trong NGỮ CẢNH, không được tự bịa thêm liều lượng, cách pha, thời gian cách ly.
"""
    elif answer_mode == "procedure":
        mode_requirements = """
- Diễn giải chi tiết từng bước, mô tả rõ mục đích của mỗi bước nếu NGỮ CẢNH có thông tin.
- Không trả lời kiểu liệt kê ngắn; mỗi bước hoặc nhóm bước nên có thêm 1–2 câu giải thích.
- Nếu NGỮ CẢNH có cảnh báo, điều kiện môi trường, hoặc lưu ý an toàn, phải đưa vào đầy đủ.
- Tuyệt đối không tự suy diễn quy trình mới ngoài những gì NGỮ CẢNH cung cấp.
"""
    elif answer_mode == "listing":
        mode_requirements = """
- Tổng hợp đầy đủ các mục xuất hiện trong NGỮ CẢNH, không bỏ sót.
- Với mỗi mục, cung cấp mô tả ngắn 1–2 câu để giải thích ý nghĩa hoặc đặc điểm chính (nếu NGỮ CẢNH có).
- Không trả lời quá ngắn theo kiểu “1. A, 2. B, 3. C”; cần mô tả rõ ràng nhưng không bịa thêm thông tin ngoài NGỮ CẢNH.
- Nếu có sự khác biệt giữa các mục (ví dụ: nhóm tác nhân, mức độ nguy hiểm, vị trí gây hại), cần chỉ rõ để người dùng hiểu sâu hơn.
"""
    else:  # general
        mode_requirements = """
- Trình bày tự nhiên nhưng chi tiết, không trả lời quá ngắn.
- Khi có nhiều ý liên quan trong NGỮ CẢNH, hãy nhóm theo chủ đề và giải thích từng ý rõ ràng.
- Mỗi ý quan trọng nên có 1–2 câu bổ sung để làm rõ cơ chế, bối cảnh hoặc ví dụ trong NGỮ CẢNH.
- Không bắt buộc theo một khung cố định, nhưng phải đảm bảo người đọc hiểu sâu vấn đề.
"""

    system_prompt = (
        "Bạn là Trợ lý Kỹ thuật Nông nghiệp & Sản phẩm của BMCVN. "
        "Luôn ưu tiên sử dụng NGỮ CẢNH được cung cấp, chỉ bổ sung kiến thức nền nông nghiệp phổ biến "
        "khi phù hợp và không mâu thuẫn với NGỮ CẢNH. "
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
- Chỉ dựa vào thông tin có trong NGỮ CẢNH, có thể gom/tổng hợp/ngắn gọn lại cho dễ hiểu.
- Không cố nhét đủ mọi mục nếu không phù hợp với câu hỏi.
- Không cần viết những câu như "Tài liệu không đề cập" trừ khi thật sự cần nhấn mạnh thiếu dữ liệu.
- Trình bày ngắn gọn, rõ ràng, có thể dùng bullet hoặc tiêu đề phụ nếu thấy cần.

Nếu phù hợp, ở cuối câu trả lời, gợi ý thêm 1–3 câu hỏi liên quan, ưu tiên lấy ý từ danh sách:
{suggestions_text}
"""

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


# ==============================
#   CHỌN SỐ DOC CHO CONTEXT
# ==============================

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

def is_listing_query(q: str) -> bool:
    t = q.lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê",
        "kể tên", "bao nhiêu", "tổng", "có bao nhiêu", "gồm",
        "các bệnh", "những bệnh", "bệnh nào", "gồm những bệnh nào"
    ])


def answer_with_suggestions(user_query: str):
    # 0. Phân luồng GLOBAL / RAG
    route = route_query(user_query)
    if route == "GLOBAL":
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia BVTV, giải thích khái niệm, viết tắt, định nghĩa một cách ngắn gọn, chuẩn sách giáo trình."
                },
                {"role": "user", "content": user_query},
            ],
        )
        return {"text": resp.choices[0].message.content.strip(), "img_keys": []}

    # 1. Chuẩn hóa query
    norm_query = normalize_query(user_query)

    # 2. Xác định dạng câu hỏi liệt kê
    is_list = is_listing_query(norm_query)

    # 3. Search
    if is_list:
        hits = search(norm_query, top_k=50)
    else:
        hits = search(norm_query, top_k=20)

    if not hits:
        return {"text": "Không tìm thấy dữ liệu phù hợp.", "img_keys": []}

    # 4. Lọc doc đủ score
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]

    if not filtered_for_main:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits[:10]
        )
        return {
            "text": "Không có tài liệu đủ độ tương đồng.\n\nGợi ý:\n" + suggestions_text,
            "img_keys": []
        }

    # 5. Quyết định số doc tối đa cho context
    max_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)

    # Ưu tiên doc include_in_context
    context_candidates = [h for h in filtered_for_main if h.get("include_in_context", False)]
    if not context_candidates:
        context_candidates = filtered_for_main

    # 6. Xác định primary_doc (ưu tiên theo mã nếu có)
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

    # 7. Build danh sách main_hits cho context
    main_hits = [primary_doc]
    for h in context_candidates:
        if h is not primary_doc and len(main_hits) < max_ctx:
            main_hits.append(h)

    # 8. Tạo context blocks
    blocks = []
    for i, h in enumerate(main_hits, 1):
        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI: {h['question']}\n"
            f"HỎI KHÁC: {h['alt_question']}\n"
            f"NỘI DUNG:\n{h['answer']}"
        )
        blocks.append(block)

    context = "\n\n--------------------\n\n".join(blocks)

    # 9. Gợi ý câu hỏi liên quan
    used = {h["question"] for h in main_hits}
    suggest = [
        h for h in hits
        if h["question"] not in used and h["score"] >= MIN_SCORE_SUGGEST
    ][:MAX_SUGGEST]

    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    # 10. Chọn answer_mode và gọi LLM sinh câu trả lời
    answer_mode = detect_answer_mode(user_query, primary_doc, is_list)
    final_answer = call_finetune_with_context(user_query, context, suggestions_text, answer_mode)

    # 11. IMG_KEY (nếu có)
    img_keys = extract_img_keys(primary_doc["answer"])

    return {"text": final_answer, "img_keys": img_keys}


# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    # q = "phân bón lá không được trộn chung với sản phẩm nào"
    import sys
    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not q:
        print("Usage: python rag_v6_merged.py <your question>")
        raise SystemExit(1)
    res = answer_with_suggestions(q)

    print("\n===== KẾT QUẢ =====\n")
    print(res["text"])

    print("\nIMG_KEY:")
    print(res["img_keys"])
