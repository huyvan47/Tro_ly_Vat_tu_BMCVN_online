import os
import time
import numpy as np
import re
import json
from openai import OpenAI

# ==============================
#       CONFIG
# ==============================

MIN_SCORE_MAIN = 0.35
MIN_SCORE_SUGGEST = 0.40
MAX_SUGGEST = 3

USE_LLM_RERANK = True      # Bật/tắt rerank bằng LLM
TOP_K_RERANK = 60          # Số doc tối đa đưa vào LLM để rerank

# ==============================
#   KB CONFIDENCE / STRATEGY
# ==============================

# KB đủ mạnh -> trả trực tiếp doc (không ép QA)
DIRECT_DOC_MIN_SCORE = 0.86
DIRECT_DOC_MIN_GAP   = 0.10    # top1 - top2

# KB phân tán/yếu -> RAG mềm (hybrid)
FRAGMENTED_MAX_TOP1  = 0.55
FRAGMENTED_MAX_GAP   = 0.03

# Giới hạn context theo mode
MAX_CTX_STRICT = 12
MAX_CTX_SOFT   = 8

# Nếu query có mã -> ưu tiên DIRECT_DOC hơn
CODE_BOOST_DIRECT = True

# ==============================
#   RERANK ROBUSTNESS
# ==============================

RERANK_SNIPPET_CHARS = 500  # cắt answer khi gửi LLM rerank
DEBUG_RERANK = False

# ==============================
#       OPENAI CLIENT
# ==============================

client = OpenAI(api_key="...")

# ==============================
#       LOAD DATA
# ==============================

data = np.load("data-kd-nam-benh-full-fix-noise.npz", allow_pickle=True)

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
    return re.findall(r'\(IMG_KEY:\s*([^)]+)\)', text or "")


def extract_codes_from_query(text: str):
    # ví dụ: cha240-06, 450-02, cha240-asmil-01...
    return re.findall(r'\b[\w]*\d[\w-]*-\d[\w-]*\b', text or "")


def analyze_hits(hits: list) -> dict:
    """
    Trả profile để quyết định chiến lược trả lời.
    """
    if not hits:
        return {"top1": 0.0, "top2": 0.0, "gap": 0.0, "mean5": 0.0, "n": 0}

    scores = [float(h.get("score", 0.0)) for h in hits]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2
    mean5 = float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores))
    return {"top1": top1, "top2": top2, "gap": gap, "mean5": mean5, "n": len(hits)}


def decide_strategy(norm_query: str, hits: list, filtered_for_main: list) -> str:
    """
    Quyết định 1 trong: DIRECT_DOC | RAG_STRICT | RAG_SOFT
    """
    prof = analyze_hits(hits)
    top1, gap = prof["top1"], prof["gap"]

    has_code = bool(extract_codes_from_query(norm_query))
    if CODE_BOOST_DIRECT and has_code and top1 >= (DIRECT_DOC_MIN_SCORE - 0.05):
        return "DIRECT_DOC"

    if top1 >= DIRECT_DOC_MIN_SCORE and gap >= DIRECT_DOC_MIN_GAP and len(filtered_for_main) > 0:
        return "DIRECT_DOC"

    if top1 <= FRAGMENTED_MAX_TOP1 or gap <= FRAGMENTED_MAX_GAP:
        return "RAG_SOFT"

    return "RAG_STRICT"


def format_direct_doc_answer(user_query: str, primary_doc: dict, suggestions_text: str) -> str:
    """
    Trả lời trực tiếp bằng nội dung tài liệu (extractive) + gợi ý.
    """
    q = (primary_doc.get("question", "") or "").strip()
    a = (primary_doc.get("answer", "") or "").strip()

    out = []
    out.append("Nội dung phù hợp nhất trong tài liệu:")

    if q:
        out.append(f"- Mục liên quan: {q}")

    if a:
        out.append("")
        out.append(a)

    if suggestions_text and suggestions_text.strip() and "(không có)" not in suggestions_text:
        out.append("")
        out.append("Câu hỏi liên quan:")
        out.append(suggestions_text)

    return "\n".join(out).strip()


def build_context_from_hits(hits_for_ctx: list) -> str:
    blocks = []
    for i, h in enumerate(hits_for_ctx, 1):
        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI: {h.get('question','')}\n"
            f"HỎI KHÁC: {h.get('alt_question','')}\n"
            f"NỘI DUNG:\n{h.get('answer','')}"
        )
        blocks.append(block)
    return "\n\n--------------------\n\n".join(blocks)


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
""".strip()
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


# ==============================
#   LLM RERANK
# ==============================

def llm_rerank(norm_query: str, results: list, top_k_rerank: int = TOP_K_RERANK):
    """
    Rerank top_k_rerank documents bằng LLM mini.
    Sử dụng include_in_context để ưu tiên doc đưa vào context.
    LUÔN trả về list hợp lệ (fallback nếu lỗi).
    """
    if not results or len(results) == 1:
        return results

    candidates = results[:top_k_rerank]

    # Build docs block
    doc_texts = []
    for i, h in enumerate(candidates):
        ans = str(h.get("answer", ""))
        if len(ans) > RERANK_SNIPPET_CHARS:
            ans = ans[:RERANK_SNIPPET_CHARS] + " ..."
        block = (
            f"[DOC {i}]\n"
            f"QUESTION: {h.get('question','')}\n"
            f"ALT_QUESTION: {h.get('alt_question','')}\n"
            f"ANSWER_SNIPPET:\n{ans}"
        )
        doc_texts.append(block)

    docs_block = "\n\n------------------------\n\n".join(doc_texts)
    if DEBUG_RERANK:
        print("docs_block:\n", docs_block)

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

    cleaned = content.replace("```json", "").replace("```", "").strip()

    try:
        ranking = json.loads(cleaned)
        if not isinstance(ranking, list):
            raise ValueError("JSON is not a list")
    except Exception:
        # fallback nếu parse fail
        return results

    idx_to_result = {i: candidates[i] for i in range(len(candidates))}

    # default
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
        except Exception:
            continue

    # reorder theo thứ tự LLM
    used = set()
    reranked = []
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if di in idx_to_result and di not in used:
                reranked.append(idx_to_result[di])
                used.add(di)
        except Exception:
            continue

    # add missing
    for i, h in enumerate(candidates):
        if i not in used:
            reranked.append(h)

    # append tail
    if len(results) > len(candidates):
        reranked.extend(results[len(candidates):])

    # ưu tiên include_in_context lên đầu
    selected = [h for h in reranked if h.get("include_in_context") is True]
    others = [h for h in reranked if not h.get("include_in_context")]

    return (selected + others) if selected else reranked


# ==============================
#        SEARCH ENGINE
# ==============================

def search(norm_query: str, top_k=40):
    vq = embed_query(norm_query)
    sims = EMBS @ vq

    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        item = {
            "question": str(QUESTIONS[i]),
            "alt_question": str(ALT_QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        }
        if CATEGORY is not None:
            item["category"] = str(CATEGORY[i])
        if TAGS is not None:
            item["tags"] = str(TAGS[i])
        results.append(item)

    if USE_LLM_RERANK and len(results) > 1:
        results = llm_rerank(norm_query, results, TOP_K_RERANK)

    return results


# ==============================
#   MODE TRẢ LỜI LINH HOẠT
# ==============================

def detect_answer_mode(user_query: str, primary_doc: dict, is_listing: bool) -> str:
    if is_listing:
        return "listing"

    text = (user_query + " " + primary_doc.get("question", "")).lower()
    cat = primary_doc.get("category", "").lower()

    if any(kw in text for kw in [
        "bệnh", "triệu chứng", "dấu hiệu", "phòng trị", "phòng trừ",
        "nứt thân", "xì mủ", "thối rễ", "cháy lá", "thán thư", "ghẻ"
    ]) or "benh" in cat:
        return "disease"

    if any(kw in text for kw in [
        "thuốc", "đặc trị", "sc", "wp", "ec", "sl",
        "dùng để làm gì", "tác dụng", "hoạt chất"
    ]) or "san_pham" in cat or "thuoc" in cat:
        return "product"

    if any(kw in text for kw in [
        "quy trình", "quy trinh", "cách làm", "hướng dẫn",
        "các bước", "làm thế nào để", "phương pháp", "thí nghiệm",
        "phân lập", "giám định", "lây bệnh trong phòng thí nghiệm"
    ]) or "quy_trinh" in cat:
        return "procedure"

    return "general"


def call_finetune_with_context(user_query, context, suggestions_text, answer_mode: str = "general", rag_mode: str = "STRICT"):
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


# ==============================
#   CHỌN SỐ DOC CHO CONTEXT
# ==============================

def choose_adaptive_max_ctx(hits_reranked, is_listing: bool = False):
    scores = [h.get("rerank_score", 0) for h in hits_reranked[:4]]
    scores += [0] * (4 - len(scores))
    s1, s2, s3, s4 = scores

    if is_listing:
        if s1 >= 0.75 and s2 >= 0.65 and s3 >= 0.55:
            return 30
        if s1 >= 0.65 and s2 >= 0.55:
            return 25
        return 20

    if s1 >= 0.90 and s2 >= 0.80 and s3 >= 0.75 and s4 >= 0.70:
        return 16
    if s1 >= 0.85 and s2 >= 0.75 and s3 >= 0.70:
        return 14
    if s1 >= 0.80 and s2 >= 0.65:
        return 12

    return 12


# ==============================
#   MAIN PIPELINE
# ==============================

def is_listing_query(q: str) -> bool:
    t = (q or "").lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê",
        "kể tên", "bao nhiêu", "tổng", "có bao nhiêu", "gồm",
        "các bệnh", "những bệnh", "bệnh nào", "gồm những bệnh nào"
    ])


def answer_with_suggestions(user_query: str):
    # 0) Route GLOBAL / RAG
    route = route_query(user_query)
    if route == "GLOBAL":
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia BVTV, giải thích khái niệm, viết tắt, định nghĩa ngắn gọn, chuẩn giáo trình."
                },
                {"role": "user", "content": user_query},
            ],
        )
        return {"text": resp.choices[0].message.content.strip(), "img_keys": []}

    # 1) Normalize query
    norm_query = normalize_query(user_query)

    # 2) Listing?
    is_list = is_listing_query(norm_query)

    # 3) Search
    hits = search(norm_query, top_k=50 if is_list else 20)
    if not hits:
        return {"text": "Không tìm thấy dữ liệu phù hợp.", "img_keys": []}

    # 4) Filter by MIN_SCORE_MAIN
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

    # 5) Decide strategy (DIRECT_DOC / RAG_STRICT / RAG_SOFT)
    strategy = decide_strategy(norm_query, hits, filtered_for_main)
    prof = analyze_hits(hits)
    print("norm_query:", norm_query)
    print("strategy:", strategy, "| profile:", prof)

    # 6) Prefer include_in_context if available
    context_candidates = [h for h in filtered_for_main if h.get("include_in_context", False)]
    if not context_candidates:
        context_candidates = filtered_for_main

    # 7) Pick primary_doc (prefer code match)
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

    # 8) Suggestions
    used_q = {primary_doc["question"]}
    suggest = [
        h for h in hits
        if h["question"] not in used_q and h["score"] >= MIN_SCORE_SUGGEST
    ][:MAX_SUGGEST]
    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    # 9) DIRECT_DOC: KB đủ mạnh -> trả trực tiếp doc (không ép QA)
    if strategy == "DIRECT_DOC":
        img_keys = extract_img_keys(primary_doc.get("answer", ""))
        text = format_direct_doc_answer(user_query, primary_doc, suggestions_text)
        return {"text": text, "img_keys": img_keys}

    # 10) Build context and call GPT answer (STRICT/SOFT)
    answer_mode = detect_answer_mode(user_query, primary_doc, is_list)

    # adaptive ctx, nhưng giới hạn theo mode
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    if strategy == "RAG_SOFT":
        max_ctx = min(MAX_CTX_SOFT, base_ctx)
        rag_mode = "SOFT"
    else:
        max_ctx = min(MAX_CTX_STRICT, base_ctx)
        rag_mode = "STRICT"

    main_hits = [primary_doc]
    for h in context_candidates:
        if h is not primary_doc and len(main_hits) < max_ctx:
            main_hits.append(h)

    context = build_context_from_hits(main_hits)

    final_answer = call_finetune_with_context(
        user_query=user_query,
        context=context,
        suggestions_text=suggestions_text,
        answer_mode=answer_mode,
        rag_mode=rag_mode,
    )

    img_keys = extract_img_keys(primary_doc.get("answer", ""))
    return {"text": final_answer, "img_keys": img_keys}


# ==============================
#   DEMO
# ==============================

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not q:
        print("Usage: python rag_kinhdoanh_v4_upgraded.py <your question>")
        raise SystemExit(1)

    res = answer_with_suggestions(q)

    print("\n===== KẾT QUẢ =====\n")
    print(res["text"])

    print("\nIMG_KEY:")
    print(res["img_keys"])
