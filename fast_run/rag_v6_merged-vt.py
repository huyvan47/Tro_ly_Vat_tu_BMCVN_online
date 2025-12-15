"""
BMCVN RAG v6: Router + QA (v4) + Verbatim (v5) in one pipeline.

Goals
- QA mode: tổng hợp/suy luận tốt (multi-doc synthesis)
- VERBATIM mode: trả đúng nguyên văn quy trình (không sai 1 ly)
- HYBRID mode: vừa tóm tắt vừa đính kèm nguyên văn (tùy chọn)

Assumptions about NPZ
- embeddings: normalized float32 vectors
- ids: chunk ids or atomic ids
- answers: chunk_text / answer text
- questions, alt_questions, category, tags exist (optional but recommended)

Security
- Do NOT hardcode API keys. Use environment variable OPENAI_API_KEY.

Test
python rag_v6_merged.py "các chai không đóng được Gardona"
python rag_v6_merged.py "quy trình làm nhãn riêng gồm những bước chính nào?"
python rag_v6_merged.py "@verbatim quy trình làm nhãn riêng"
python rag_v6_merged.py "@qa quy trình làm nhãn riêng gồm những bước chính nào?"
"""

import os
import re
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

from openai import OpenAI


# =========================
# CONFIG
# =========================

NPZ_PATH = "data-vat-tu-full-merge-overlap.npz"

EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"

# Retrieval thresholds (tune)
MIN_SCORE_MAIN = 0.35
MIN_SCORE_SUGGEST = 0.40
MAX_SUGGEST = 3

# Candidate pools
TOPK_QA = 20
TOPK_LISTING = 50

# LLM rerank
USE_LLM_RERANK = True
TOP_K_RERANK = 60

# Router signals
TOPK_ROUTER = 10              # used to compute parent concentration
PARENT_DOMINANCE_PCT = 0.60   # >= 60% of topK share same parent -> likely SOP single doc
LOW_CONF_TOP1 = 0.33          # top1 below this often means weak match
LOW_MARGIN = 0.03             # top1-top2 small means ambiguous

# Verbatim export
MAX_SOURCE_CHARS_PER_CALL = 12000   # paging when exporting full parent
VERBATIM_USE_GPT = False            # safest: print directly, no LLM

# =========================
# CLIENT
# =========================

client = OpenAI(api_key="...")


# =========================
# LOAD NPZ
# =========================

data = np.load(NPZ_PATH, allow_pickle=True)

EMBS = data["embeddings"].astype(np.float32)  # assumed normalized
IDS = data.get("ids", data.get("id")).astype(object)
ANSWERS = data["answers"].astype(object)

QUESTIONS = data.get("questions", None)
ALT_QUESTIONS = data.get("alt_questions", None)
CATEGORY = data.get("category", None)
TAGS = data.get("tags", None)


# =========================
# BASIC NORMALIZATION
# =========================

def normalize_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_parent_and_index(doc_id: str) -> Tuple[str, int]:
    """
    Expect: <parent>_chunk_<NN>
    If not match -> (doc_id, 0) (atomic doc)
    """
    s = str(doc_id)
    m = re.match(r"^(.*?)_chunk_(\d+)$", s)
    if not m:
        return s, 0
    return m.group(1), int(m.group(2))


def is_listing_query(q: str) -> bool:
    t = (q or "").lower()
    return any(x in t for x in [
        "các loại", "những loại", "bao nhiêu loại", "tất cả", "liệt kê", "kể tên",
        "bao nhiêu", "tổng", "có bao nhiêu", "gồm"
    ])


def remove_img_keys(text: str) -> str:
    return re.sub(r"-?\s*\(IMG_KEY:[^)]+\)\s*", "", str(text)).strip()


def extract_img_keys(text: str) -> List[str]:
    return re.findall(r"\(IMG_KEY:\s*([^)]+)\)", str(text))


def extract_codes_from_query(text: str) -> List[str]:
    # your original pattern
    return re.findall(r"\b[\w]*\d[\w-]*-\d[\w-]*\b", str(text))


# =========================
# EMBEDDING
# =========================

def embed_query(q: str) -> np.ndarray:
    """
    IMPORTANT: keep the same embedding format as doc embedding build script.
    Recommended query format: "Q: <query>"
    """
    q = normalize_ws(q)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[f"Q: {q}"])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def cosine_scores(vq: np.ndarray) -> np.ndarray:
    return EMBS @ vq


# =========================
# CATEGORY / TAG BOOST (same spirit as v4)
# =========================

def detect_query_category(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["misa", "mua hàng", "nhập kho", "hóa đơn"]):
        return "quy_trinh_mua_hang_misa"
    if any(k in t for k in ["kế hoạch", "dự trù", "vật tư", "kế hoạch sử dụng"]):
        return "ke_hoach_vat_tu"
    if any(k in t for k in ["chai", "pet", "hdpe", "nhôm", "nhom"]):
        return "chai"
    if any(k in t for k in ["thùng", "carton", "thung"]):
        return "thung"
    if any(k in t for k in ["tem", "nhãn", "màng", "túi", "bao bì", "goi", "nhan"]):
        return "nhan_mang_tui"
    return None


def boost_similarity_scores(raw_sims: np.ndarray, query_text: str) -> np.ndarray:
    sims = raw_sims.copy()
    qcat = detect_query_category(query_text)
    qtext = (query_text or "").lower()

    if CATEGORY is None and TAGS is None:
        return sims

    for i in range(len(sims)):
        # boost category
        if CATEGORY is not None and qcat is not None and str(CATEGORY[i]) == qcat:
            sims[i] += 0.12
        # boost tags
        if TAGS is not None:
            taglist = str(TAGS[i]).lower().split("|")
            if any(t.strip() and t.strip() in qtext for t in taglist):
                sims[i] += 0.05
    return sims


# =========================
# LLM RERANK (same as v4 but slightly safer)
# =========================

def llm_rerank(norm_query: str, results: List[Dict[str, Any]], top_k_rerank: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
    if not results or len(results) == 1:
        return results

    candidates = results[:top_k_rerank]
    doc_texts = []
    for i, h in enumerate(candidates):
        ans = str(h.get("answer", ""))
        if len(ans) > 500:
            ans = ans[:500] + " ..."
        doc_texts.append(
            f"[DOC {i}]\nQUESTION: {h.get('question','')}\nALT_QUESTION: {h.get('alt_question','')}\nANSWER_SNIPPET:\n{ans}"
        )

    docs_block = "\n\n------------------------\n\n".join(doc_texts)

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

    try:
        resp = client.chat.completions.create(
            model=RERANK_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        cleaned = (resp.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
        ranking = json.loads(cleaned) if cleaned else []
        if not isinstance(ranking, list):
            return results
    except Exception:
        return results

    # attach rerank scores
    for h in candidates:
        h["rerank_score"] = 0.0
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            sc = float(item.get("score", 0))
            if 0 <= di < len(candidates):
                candidates[di]["rerank_score"] = sc
        except Exception:
            continue

    # reorder by provided list, then append remaining
    used = set()
    reranked = []
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if 0 <= di < len(candidates) and di not in used:
                reranked.append(candidates[di]); used.add(di)
        except Exception:
            continue
    for i, h in enumerate(candidates):
        if i not in used:
            reranked.append(h)
    if len(results) > len(candidates):
        reranked.extend(results[len(candidates):])
    return reranked


# =========================
# SEARCH (shared)
# =========================

def search(norm_query: str, top_k: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    vq = embed_query(norm_query)
    raw = cosine_scores(vq)
    sims = boost_similarity_scores(raw, norm_query)
    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        results.append({
            "id": str(IDS[i]),
            "question": str(QUESTIONS[i]) if QUESTIONS is not None else "",
            "alt_question": str(ALT_QUESTIONS[i]) if ALT_QUESTIONS is not None else "",
            "answer": str(ANSWERS[i]),
            "category": str(CATEGORY[i]) if CATEGORY is not None else "",
            "tags": str(TAGS[i]) if TAGS is not None else "",
            "score": float(sims[i]),
        })

    if USE_LLM_RERANK and len(results) > 1:
        results = llm_rerank(norm_query, results, TOP_K_RERANK)

    return results, idx, sims[idx]


# =========================
# ROUTER (v4 + v5 merge)
# =========================

def forced_mode_prefix(user_query: str) -> Optional[str]:
    q = (user_query or "").strip().lower()
    if q.startswith("@verbatim"):
        return "VERBATIM"
    if q.startswith("@qa"):
        return "QA"
    if q.startswith("@hybrid"):
        return "HYBRID"
    return None


def detect_mode(user_query: str, hits_router: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Decide mode based on:
    - keyword intent
    - parent concentration among top-k
    - confidence (top1 score, margin)
    - user force prefix
    """
    forced = forced_mode_prefix(user_query)
    if forced:
        return forced, {"reason": "forced_prefix"}

    q = (user_query or "").lower()

    # signals from retrieval
    top1 = hits_router[0]["score"] if hits_router else 0.0
    top2 = hits_router[1]["score"] if len(hits_router) > 1 else 0.0
    margin = top1 - top2

    parents = [parse_parent_and_index(h["id"])[0] for h in hits_router]
    cnt = Counter(parents)
    parent_id, parent_hits = cnt.most_common(1)[0] if parents else ("", 0)
    dominance = (parent_hits / max(1, len(hits_router)))

    # intent keywords
    verbatim_kw = [
        "nguyên văn", "đầy đủ", "100%", "không sai", "đúng từng chữ",
        "quy trình", "các bước", "hướng dẫn", "sop", "biểu mẫu", "hồ sơ",
        "trình tự", "workflow", "quy tắc", "checklist", "bước", "những bước" 
    ]
    wants_verbatim = any(k in q for k in verbatim_kw)

    qa_kw = ["vì sao", "tại sao", "nguyên nhân", "không đóng được", "không được", "so sánh", "khác nhau", "nên", "khuyến nghị"]
    wants_qa = any(k in q for k in qa_kw) or is_listing_query(q)

    # decision
    # 1) If user wants verbatim OR retrieval strongly points to single parent SOP
    if wants_verbatim or dominance >= PARENT_DOMINANCE_PCT:
        # If query also asks analysis ("vì sao", "nguyên nhân"), pick HYBRID
        if any(k in q for k in ["vì sao", "tại sao", "nguyên nhân", "rủi ro", "lưu ý"]) and wants_verbatim:
            return "HYBRID", {"reason": "verbatim_intent_with_analysis", "parent": parent_id, "dominance": dominance, "top1": top1, "margin": margin}
        return "VERBATIM", {"reason": "verbatim_or_parent_dominant", "parent": parent_id, "dominance": dominance, "top1": top1, "margin": margin}

    # 2) If QA intent -> QA
    if wants_qa:
        return "QA", {"reason": "qa_intent", "parent": parent_id, "dominance": dominance, "top1": top1, "margin": margin}

    # 3) If low confidence or ambiguous -> QA (with possibility to offer verbatim)
    if top1 < LOW_CONF_TOP1 or margin < LOW_MARGIN:
        return "QA", {"reason": "low_conf_or_ambiguous", "parent": parent_id, "dominance": dominance, "top1": top1, "margin": margin}

    # default QA
    return "QA", {"reason": "default", "parent": parent_id, "dominance": dominance, "top1": top1, "margin": margin}


# =========================
# VERBATIM ENGINE (v5 improved: weighted parent vote)
# =========================

def choose_parent_by_weighted_vote(hits: List[Dict[str, Any]]) -> str:
    """
    Sum scores per parent, choose max.
    More stable than majority vote.
    """
    s = defaultdict(float)
    for h in hits:
        p, _ = parse_parent_and_index(h["id"])
        s[p] += float(h.get("score", 0.0))
    return max(s.items(), key=lambda x: x[1])[0] if s else ""


def fetch_all_chunks_by_parent(parent_id: str) -> List[Tuple[int, str, str]]:
    prefix = str(parent_id) + "_chunk_"
    items = []
    for i, cid in enumerate(IDS):
        s = str(cid)
        if s.startswith(prefix):
            _, cidx = parse_parent_and_index(s)
            items.append((cidx, s, str(ANSWERS[i])))
    items.sort(key=lambda x: x[0])
    return items


def paginate_chunks(items: List[Tuple[int, str, str]], max_chars: int = MAX_SOURCE_CHARS_PER_CALL):
    pages, cur, cur_len = [], [], 0
    for cidx, cid, txt in items:
        block_len = len(cid) + len(txt) + 20
        if cur and cur_len + block_len > max_chars:
            pages.append(cur); cur=[]; cur_len=0
        cur.append((cidx, cid, txt))
        cur_len += block_len
    if cur:
        pages.append(cur)
    return pages


def verbatim_export(user_query: str, hits_router: List[Dict[str, Any]]) -> Dict[str, Any]:
    parent_id = choose_parent_by_weighted_vote(hits_router[:TOPK_ROUTER])

    # If the chosen parent has no chunks, fallback to best single doc (atomic)
    chunks = fetch_all_chunks_by_parent(parent_id)
    if not chunks:
        # just return top doc answer verbatim
        best = hits_router[0] if hits_router else None
        if not best:
            return {"mode": "VERBATIM", "parent": None, "text": "KHÔNG TÌM THẤY TRONG NGUỒN", "img_keys": []}
        return {
            "mode": "VERBATIM",
            "parent": parse_parent_and_index(best["id"])[0],
            "text": best["answer"],
            "img_keys": extract_img_keys(best["answer"]),
        }

    # safest: print directly
    pages = paginate_chunks(chunks)
    out_parts = []
    for pageno, page_items in enumerate(pages, start=1):
        text = "\n\n".join([txt for _, _, txt in page_items]).strip()
        out_parts.append(f"=== PART {pageno}/{len(pages)} | PARENT {parent_id} ===\n{text}")

    return {"mode": "VERBATIM", "parent": parent_id, "text": "\n\n".join(out_parts), "img_keys": []}


# =========================
# QA ENGINE (v4)
# =========================

def choose_adaptive_max_ctx(hits_reranked: List[Dict[str, Any]], is_listing: bool = False) -> int:
    scores = [h.get("rerank_score", 0.0) for h in hits_reranked[:4]]
    scores += [0.0] * (4 - len(scores))
    s1, s2, s3, s4 = scores

    if is_listing:
        if s1 >= 0.75 and s2 >= 0.65 and s3 >= 0.55:
            return 25
        if s1 >= 0.65 and s2 >= 0.55:
            return 20
        return 15

    if s1 >= 0.90 and s2 >= 0.80 and s3 >= 0.75 and s4 >= 0.70:
        return 14
    if s1 >= 0.85 and s2 >= 0.75 and s3 >= 0.70:
        return 12
    if s1 >= 0.80 and s2 >= 0.65:
        return 10
    return 10


def call_answer_llm(user_query: str, context: str, suggestions_text: str) -> str:
    system_prompt = (
        "Bạn là Trợ lý Vật tư BMCVN. Trả lời dựa hoàn toàn trên NGỮ CẢNH.\n"
        "- Tổng hợp đầy đủ các ý liên quan trong NGỮ CẢNH.\n"
        "- Không bịa. Nếu NGỮ CẢNH không nói tới phần nào, nêu rõ 'tài liệu không đề cập'.\n"
        "- Khi trả lời về quy trình/SOP: ưu tiên bám sát câu chữ tài liệu, không tự suy diễn.\n"
        "- Nếu câu hỏi yêu cầu 'nguyên văn/100%' thì trả theo nguyên văn (không diễn giải).\n"
    )

    user_prompt = f"""
NGỮ CẢNH:
\"\"\"{context}\"\"\"

CÂU HỎI:
\"\"\"{user_query}\"\"\"

YÊU CẦU:
- Trả lời chi tiết, rõ ràng; ưu tiên bullet.
- Sử dụng TẤT CẢ thông tin liên quan trong NGỮ CẢNH (không chỉ 1 DOC).
- Không đổi mã chai/mã vật tư.
- Gợi ý thêm 1–3 câu hỏi liên quan từ danh sách:
{suggestions_text}
""".strip()

    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        max_completion_tokens=1500,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def qa_answer(user_query: str) -> Dict[str, Any]:
    norm_query = normalize_ws(user_query)

    top_k = TOPK_LISTING if is_listing_query(norm_query) else TOPK_QA
    hits, _, _ = search(norm_query, top_k=top_k)

    if not hits:
        return {"mode": "QA", "text": "Không tìm thấy dữ liệu phù hợp.", "img_keys": []}

    # filter main
    filtered_for_main = [h for h in hits if h["score"] >= MIN_SCORE_MAIN]
    if not filtered_for_main:
        suggestions_text = "\n".join(f"- {h['question']} (score={h['score']:.2f})" for h in hits[:10])
        return {"mode": "QA", "text": "Không có tài liệu đủ độ tương đồng.\n\nGợi ý:\n" + suggestions_text, "img_keys": []}

    # primary doc: prioritize codes
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

    # build context (diversify by parent to avoid repeating same SOP chunks too much)
    is_list = is_listing_query(norm_query)
    max_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)

    main_hits = [primary_doc]
    used_parents = {parse_parent_and_index(primary_doc["id"])[0]}
    for h in filtered_for_main:
        if h is primary_doc:
            continue
        if len(main_hits) >= max_ctx:
            break
        p = parse_parent_and_index(h["id"])[0]
        # light diversification: allow at most 3 per parent unless listing
        if not is_list:
            per_parent = sum(1 for x in main_hits if parse_parent_and_index(x["id"])[0] == p)
            if per_parent >= 3:
                continue
        main_hits.append(h)

    blocks = []
    for i, h in enumerate(main_hits, 1):
        clean_ans = remove_img_keys(h["answer"])
        blocks.append(
            f"[DOC {i} | id={h['id']} | score={h['score']:.3f}]\n"
            f"CÂU HỎI: {h.get('question','')}\n"
            f"HỎI KHÁC: {h.get('alt_question','')}\n"
            f"NỘI DUNG:\n{clean_ans}"
        )
    context = "\n\n--------------------\n\n".join(blocks)

    used_q = {h.get("question","") for h in main_hits}
    suggest = [h for h in hits if h.get("question","") not in used_q and h["score"] >= MIN_SCORE_SUGGEST][:MAX_SUGGEST]
    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    final = call_answer_llm(user_query, context, suggestions_text)
    img_keys = extract_img_keys(primary_doc["answer"])
    return {"mode": "QA", "text": final, "img_keys": img_keys}


# =========================
# HYBRID ENGINE (QA summary + verbatim appendix when SOP-like)
# =========================

def hybrid_answer(user_query: str) -> Dict[str, Any]:
    # 1) QA answer for comprehension
    qa = qa_answer(user_query)

    # 2) Verbatim appendix
    norm_query = normalize_ws(user_query)
    hits_router, _, _ = search(norm_query, top_k=max(TOPK_ROUTER, 15))
    vb = verbatim_export(user_query, hits_router)

    text = (
        "PHẦN 1) TÓM TẮT / GIẢI THÍCH (QA)\n"
        "--------------------------------\n"
        f"{qa['text']}\n\n"
        "PHẦN 2) NGUYÊN VĂN THEO TÀI LIỆU (VERBATIM)\n"
        "------------------------------------------\n"
        f"{vb['text']}"
    )
    return {"mode": "HYBRID", "parent": vb.get("parent"), "text": text, "img_keys": qa.get("img_keys", [])}


# =========================
# MAIN ENTRYPOINT
# =========================

def answer(user_query: str) -> Dict[str, Any]:
    norm_query = normalize_ws(user_query)

    # router uses a small hit set
    hits_router, _, _ = search(norm_query, top_k=max(TOPK_ROUTER, 15))
    mode, debug = detect_mode(user_query, hits_router)

    if mode == "VERBATIM":
        out = verbatim_export(user_query, hits_router)
        out["router_debug"] = debug
        return out

    if mode == "HYBRID":
        out = hybrid_answer(user_query)
        out["router_debug"] = debug
        return out

    out = qa_answer(user_query)
    out["router_debug"] = debug
    return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not q:
        print("Usage: python rag_v6_merged.py <your question>")
        raise SystemExit(1)
    res = answer(q)
    print("\n=== MODE ===")
    print(res.get("mode"))
    print("\n=== ROUTER DEBUG ===")
    print(res.get("router_debug"))
    print("\n=== ANSWER ===\n")
    print(res.get("text", ""))
    if res.get("img_keys"):
        print("\nIMG_KEY:", res["img_keys"])
