from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query, extract_img_keys
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.answer_modes import decide_answer_policy
from rag.generator import call_finetune_with_context
from rag.tag_filter import tag_filter_pipeline
from rag.logging.timing_logger import TimingLog
from rag.reasoning.multi_hop import multi_hop_controller
from typing import List, Tuple, Dict, Any
from rag.logging.debug_log import debug_log
from rag.post_answer.enricher import enrich_answer_if_needed

FORCE_MUST_TAGS = {
    "mechanisms:luu-dan-manh",
    "mechanisms:luu-dan",
    "mechanisms:tiep-xuc-manh",
    "mechanisms:tiep-xuc",
    "mechanisms:tiep-xuc-luu-dan-manh",
    "mechanisms:tiep-xuc-luu-dan",
    "mechanisms:xong-hoi-manh",
    "mechanisms:xong-hoi",
    "mechanisms:co-chon-loc",
    "mechanisms:khong-chon-loc",
}

FORMULA_TRIGGERS = [
    
    "công thức",
    "phối trộn",
    "phối hợp thuốc",
    "liều phối",
    "phối",
    "pha thuốc",
    "công thức trị",
    "công thức trừ",
    "công thức diệt",
    "phác đồ",
    "kết hợp thuốc",
    "hoạt chất lưu dẫn phù hợp",
]


def is_formula_query(query: str, tags: dict) -> bool:
    """
    Nhận diện truy vấn dạng phối công thức.
    """

    has_plus = "+" in query
    has_mechanisms = any(
        t.startswith("mechanisms:")
        for t in tags.get("must", []) + tags.get("soft", [])
    )

    # Nếu có nhiều hơn 1 mechanism tag -> gần như chắc chắn là phối
    num_mechs = sum(
        1 for t in tags.get("must", []) + tags.get("soft", [])
        if t.startswith("mechanisms:")
    )

    if has_mechanisms and (has_plus or num_mechs >= 2):
        return True

    return False

def formula_mode_search(
    *,
    client,
    kb,
    norm_query: str,
    must_tags: List[str],
):
    """
    Tìm kiếm theo chế độ công thức (không dùng multi-hop).
    Budget-aware:
    - Tổng ngân sách = RAGConfig.max_ctx_soft
    - Chia đều cho mỗi must_tag
    - Nếu chưa đủ → chạy free search để bù
    """

    max_ctx = RAGConfig.max_ctx_soft

    must_tags = list(must_tags or [])
    num_tags = max(len(must_tags), 1)

    # ngân sách cho mỗi tag
    per_tag_k = max(1, max_ctx // num_tags)

    all_results = []

    # ---- ROLE 1: MUST TAG (chia ngân sách) ----
    for m in must_tags:
        print("m:", m)
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=per_tag_k,
            must_tags=[m],
            any_tags=[]
        )
        all_results.extend(hits)

    # dedupe theo id
    unique = {}
    for h in all_results:
        hid = h.get("id")
        if hid:
            prev = unique.get(hid)
            if not prev or h.get("score", 0) > prev.get("score", 0):
                unique[hid] = h

    results = list(unique.values())

    # ---- ROLE 2: FREE SEARCH (bù slot còn thiếu) ----
    remaining = max_ctx - len(results)

    if remaining > 0:
        hits_free = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=remaining,
            must_tags=[],
            any_tags=[]
        )

        for h in hits_free:
            hid = h.get("id")
            if hid and hid not in unique:
                unique[hid] = h
                if len(unique) >= max_ctx:
                    break

        results = list(unique.values())

    # hard cap an toàn
    return results[:max_ctx]

def preserve_search_order(hits):
    """
    Đánh dấu thứ tự gốc từ search() để pipeline KHÔNG làm xáo trộn.
    """
    for idx, h in enumerate(hits):
        h["_search_rank"] = idx
    return hits

def _count_tag_hits(h, any_tags, must_tags):
    tv2 = str(h.get("tags_v2") or "")
    score = 0
    for t in (must_tags or []):
        if t and t in tv2:
            score += 3
    for t in (any_tags or []):
        if t and t in tv2:
            score += 1
    return score

def _global_system_prompt() -> str:
    return """
Bạn là chuyên gia BVTV/nông học tại Việt Nam. Mục tiêu: cung cấp câu trả lời CHẤT LƯỢNG CAO theo phong cách giáo trình/chuyên khảo,
giải thích rõ ràng, có chiều sâu, giàu ví dụ thực tế trong canh tác Việt Nam.

TIÊU CHUẨN CHẤT LƯỢNG (BẮT BUỘC):
- Ưu tiên: chính xác, mạch lạc, có tính “giải thích được” (explainable), không nói chung chung.
- Trình bày theo cấu trúc rõ ràng, có tiêu đề; dùng bullet và bảng (nếu hữu ích).
- Luôn phân biệt: (i) điều chắc chắn/phổ quát, (ii) điều phụ thuộc bối cảnh (cây, giai đoạn, thời tiết, áp lực dịch hại), (iii) điều cần thêm dữ liệu.
- Khi thuật ngữ/đối tượng có nhiều cách gọi tại VN: nêu tên thường gọi + mô tả nhận diện; tránh bịa tên loài.
- Nếu thiếu dữ liệu để kết luận chắc: nói rõ “phụ thuộc/ cần xác minh” và đưa tiêu chí/quan sát để người dùng tự kiểm chứng.

CẤU TRÚC CÂU TRẢ LỜI CHUẨN:
1) Tóm tắt nhanh (2–4 dòng): trả lời trực diện câu hỏi.
2) Định nghĩa/khái niệm cốt lõi (ngắn gọn).
3) Đặc điểm nhận biết / điểm then chốt (3–7 bullet).
4) Cơ chế / nguyên lý (nếu liên quan): giải thích ở mức vừa đủ, tránh thuật ngữ quá hàn lâm nhưng phải đúng.
5) Phân loại (CHỈ KHI câu hỏi hỏi “gồm những loại nào/bao gồm/phân loại”): kèm tiêu chí phân biệt.
6) Ví dụ đại diện: ưu tiên nhóm/case phổ biến trong canh tác Việt Nam (nêu 3–8 ví dụ phù hợp).
7) Sai lầm thường gặp & cách tránh (2–5 ý) — chỉ nêu khi giúp ích trực tiếp.
8) Câu hỏi cần làm rõ (2–6 câu): để chốt quyết định thực tế theo bối cảnh người dùng.

QUY TẮC TRẢ LỜI:
- Không lan man sang chủ đề ngoài trọng tâm câu hỏi.
- Không “tỏ ra chắc chắn” khi thiếu cơ sở; không suy diễn vượt quá thông tin đầu vào.
- Dùng thuật ngữ BVTV quen thuộc tại Việt Nam; nếu dùng thuật ngữ quốc tế thì giải thích ngắn kèm theo.
- Văn phong chuyên nghiệp, dễ hiểu; ưu tiên ví dụ và tiêu chí phân biệt hơn là lý thuyết dài dòng.
""".strip()

def answer_with_suggestions(*, user_query, kb, client, cfg, policy):
    timer = TimingLog(user_query)
    # -----------------------------------------------------
    # 0) ROUTER – QUYỀN CAO NHẤT
    # -----------------------------------------------------
    route = route_query(client, user_query)

    norm_query = normalize_query(client, user_query)
    norm_lower = norm_query.lower()
    timer.mark("normalize")

    force_rag = any(k in norm_lower for k in FORMULA_TRIGGERS)

    if force_rag:
        route = "RAG"

    print("route:", route)
    timer.mark("router")

    # -----------------------------------------------------
    # 1) NHÁNH GLOBAL NGAY TỪ ĐẦU (chỉ khi router cho phép)
    # -----------------------------------------------------
    if route == "GLOBAL":
        model = "gpt-4.1"
        resp = client.chat.completions.create(
            model=model,
            # temperature=0.25 if hard else 0.35,
            # max_completion_tokens=3500 if hard else 2500,
            temperature=0.25 ,
            max_completion_tokens=3500,
            messages=[
                {"role": "system", "content": _global_system_prompt()},
                {"role": "user", "content": user_query},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return {
            "text": text,
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": "",
            "strategy": f"GLOBAL/{model}",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # -----------------------------------------------------
    # 2) NORMALIZE + TAGS
    # -----------------------------------------------------

    is_list = is_listing_query(norm_query)

    # LUÔN chạy tag_filter_pipeline để đảm bảo có result
    result = tag_filter_pipeline(norm_query)
    timer.mark("tag_filter")

    must_tags = result.get("must", [])
    any_tags = result.get("any", [])

    if is_formula_query(norm_query, result):

        print("[MODE] Formula-based retrieval")

        hits = formula_mode_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            must_tags=must_tags,
        )

    else:

        print("[MODE] Knowledge multi-hop retrieval")

        hits = multi_hop_controller(
            client=client,
            kb=kb,
            base_query=norm_query,
            must_tags=must_tags,
            any_tags=any_tags,
            timer=timer,
        )

    # answer_mode = result.get("answer_mode", "")

    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)
    debug_log("QUERY      :", norm_query)
    debug_log("MUST TAGS  :", must_tags)
    debug_log("ANY TAGS   :", any_tags)

    # -----------------------------------------------------
    # 5) RAG PIPELINE – SINGLE hoặc MULTI QUERY
    # -----------------------------------------------------

    hits = preserve_search_order(hits)

    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "NO_HITS",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    for h in hits:
        # chỉ để phân tích / debug / profile
        h["fused_score"] = fused_score(h)
        h["tag_hits"] = _count_tag_hits(h, any_tags, must_tags)

    primary_doc = hits[0]

    # -----------------------------------------------------
    # 8) BUILD CONTEXT + GENERATE
    # -----------------------------------------------------
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)

    context = build_context_from_hits(hits[:max_ctx])
    timer.mark("build_context")

    policy = decide_answer_policy(user_query, primary_doc, force_listing=is_list)
    answer_mode = "listing" if policy.format == "listing" else policy.intent
    if policy.format == "listing":
        answer_mode_final = "listing"
    else:
        answer_mode_final = answer_mode
    final_answer = call_finetune_with_context(
        client=client,
        user_query=user_query,
        context=context,
        answer_mode=answer_mode_final,
        rag_mode="STRICT",
    )
    timer.mark("llm_generate")

    final_answer = enrich_answer_if_needed(
        client=client,
        user_query=user_query,
        answer_text=final_answer,
        answer_mode=answer_mode_final,
        any_tags=any_tags,
        must_tags=must_tags,
        route="RAG",
    )

    timer.mark("enrich_answer_if_needed")

    img_keys = extract_img_keys(primary_doc.get("answer", ""))

    timer.finish(RAGConfig.enable_timing_log)
    return {
        "text": final_answer,
        "img_keys": img_keys,
        "route": "RAG",
        "norm_query": norm_query,
        # "strategy": strategy,
        # "profile": prof,
        "context_build": context,
    }
