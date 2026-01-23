from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query, extract_img_keys
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score, analyze_hits_fused
from rag.strategy import decide_strategy
from rag.text_utils import extract_codes_from_query
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.answer_modes import decide_answer_policy
from rag.formatter import format_direct_doc_answer
from rag.generator import call_finetune_with_context
from rag.logging.multi_query_logger import write_multi_query_logs
# from rag.verbatim import verbatim_export
from rag.tag_filter import tag_filter_pipeline
from rag.logging.timing_logger import TimingLog
from rag.reasoning.multi_hop import multi_hop_controller
from typing import List, Tuple, Dict, Any
from rag.logging.debug_log import debug_log
from rag.post_answer.enricher import enrich_answer_if_needed
import unicodedata
import re
import json
from typing import Dict, Any


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
    "pha thuốc",
    "công thức trị",
    "công thức trừ",
    "công thức diệt",
    "phác đồ",
    "liều phối",
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
    any_tags: List[str],
    top_k: int
):
    """
    Tìm kiếm theo chế độ công thức (không dùng multi-hop).
    """

    all_results = []

    # ---- ROLE 1: MUST TAG ----
    for m in must_tags:
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=top_k,
            must_tags=[m],
            any_tags=[]
        )
        all_results.extend(hits)

    # # ---- ROLE 2: SOFT TAG ----
    # for s in any_tags:
    #     hits = retrieve_search(
    #         client=client,
    #         kb=kb,
    #         norm_query=norm_query,
    #         top_k=top_k,
    #         must_tags=[s],
    #         any_tags=[]
    #     )
    #     all_results.extend(hits)

    # Dedupe theo id
    unique = {}
    for h in all_results:
        hid = h.get("id")
        if hid:
            unique[hid] = h

    return list(unique.values())

def multi_query_retrieve(
    *,
    client,
    kb,
    norm_query: str,
    must_tags: List[str],
    any_tags: List[str],
    answer_mode_hint: str,
    timer=None,
):
    """
    Thực hiện:
      llm_build_sub_queries →
      retrieve từng sub →
      weighted_rrf_fuse
    """

    subs = llm_build_sub_queries(
        client=client,
        norm_query=norm_query,
        must_tags=must_tags,
        any_tags=any_tags,
        answer_mode_hint=answer_mode_hint,
        max_variants=RAGConfig.max_sub_queries,
    )

    if not subs:
        return None

    results_by_query = []

    for qi, sub in enumerate(subs):
        q = sub["q"]
        purpose = sub.get("purpose", "general")

        hits_i = retrieve_search(
            client=client,
            kb=kb,
            norm_query=q,
            top_k=RAGConfig.multi_query_top_k,
            must_tags=must_tags,
            any_tags=any_tags,
        )

        timer = TimingLog(norm_query)
        # Ghi log thời gian từng sub-query
        timer.mark("tag_filter")

        results_by_query.append({
            "purpose": purpose,
            "qi": qi,
            "weight": _purpose_weight(purpose),
            "hits": hits_i or [],
        })

    fused = _weighted_rrf_fuse(
        results_by_query,
        k=RAGConfig.rrf_k,
        top_n=RAGConfig.rrf_top_n,
    )

    if RAGConfig.enable_multi_query_log:
        write_multi_query_logs(
            original_query=norm_query,
            subs=subs,
            results_by_query=results_by_query,
            fused_hits=fused,
        )

    return fused

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

def promote_forced_tags(must_tags, any_tags):
    must = set(must_tags or [])
    anyt = set(any_tags or [])

    forced = anyt & FORCE_MUST_TAGS
    if forced:
        must |= forced
        anyt -= forced

    return list(must), list(anyt)

_space_re = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("đ", "d") 
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = _space_re.sub(" ", s)
    return s


def analyze_intent_and_slots(
    *,
    client,
    norm_query: str,
    any_tags: List[str],
    max_tokens: int = 450,
) -> Dict[str, Any]:
    """
    Trả về JSON:
      intent_type: global_pure | global_conditional | rag_catalog
      required_slots: [...]
      missing_slots: [...]
      should_ask_back: bool
      ask_back_questions: [<=2 câu]
      route_override: GLOBAL | RAG | ""
      answer_mode_override: GLOBAL_BOUNDED | ""  (tuỳ bạn)
      confidence: 0..1
      reasons: [..]
    """
    sys = (
        "Bạn là bộ phân tích intent/slot cho hệ thống tư vấn BVTV.\n"
        "Nhiệm vụ: phân loại câu hỏi và xác định thông tin tối thiểu để trả lời CHẮC.\n"
        "QUY TẮC:\n"
        "- KHÔNG trả lời nội dung BVTV.\n"
        "- Chỉ trả JSON hợp lệ.\n"
        "- Ưu tiên an toàn: nếu thiếu thông tin quan trọng (PHI/cây trồng/sản phẩm/hoạt chất/thời điểm) "
        "thì should_ask_back=true.\n"
        "- ask_back_questions tối đa 2 câu, ngắn, dễ trả lời.\n"
        "\n"
        "Gợi ý phân loại:\n"
        "- global_pure: câu hỏi nguyên lý chung (vd: diệt nấm tận gốc?)\n"
        "- global_conditional: nhìn như chung nhưng phụ thuộc biến (vd: gần thu hoạch an toàn không?)\n"
        "- rag_catalog: hỏi danh mục/sản phẩm/phối trộn cụ thể.\n"
    )

    payload = {
        "norm_query": (norm_query or "").strip(),
        "any_tags": any_tags or [],
        "output_schema": {
            "intent_type": "global_pure|global_conditional|rag_catalog",
            "required_slots": ["string"],
            "missing_slots": ["string"],
            "should_ask_back": "boolean",
            "ask_back_questions": ["string"],
            "route_override": "GLOBAL|RAG|",
            "answer_mode_override": "GLOBAL_BOUNDED|",
            "confidence": "number",
            "reasons": ["string"]
        },
        "slot_hints": [
            "crop (cây trồng)", "days_to_harvest (còn bao nhiêu ngày thu hoạch)",
            "product_or_ai (tên sản phẩm/hoạt chất)", "phi_label (PHI theo nhãn)",
            "pest_or_disease (đối tượng)", "application_method (phun/tưới/rải)"
        ]
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        # sanitize nhỏ
        data["ask_back_questions"] = (data.get("ask_back_questions") or [])[:2]
        data["required_slots"] = data.get("required_slots") or []
        data["missing_slots"] = data.get("missing_slots") or []
        return data
    except Exception:
        return {}

def llm_build_sub_queries(
    *,
    client,
    norm_query: str,
    must_tags: List[str],
    any_tags: List[str],
    answer_mode_hint: str = "",
    max_variants: int = 5,
) -> List[Dict[str, str]]:
    """
    Dùng LLM để tạo sub-queries. Trả về list dict: {purpose, q}
    - Không tạo tags mới (tags đã có).
    - Output JSON ổn định để parse.
    """
    sys = (
        "Bạn là module tạo sub-query cho hệ thống RAG nông nghiệp.\n"
        "Nhiệm vụ: sinh các câu truy vấn ngắn, dễ match tài liệu, KHÔNG trả lời người dùng.\n"
        "Yêu cầu:\n"
        "- Chỉ trả về JSON hợp lệ.\n"
        "- Mỗi sub-query 100% là tiếng Việt (trừ tên hoạt chất)."
        "- Mỗi sub-query phải đánh vào một góc khác nhau: hoạt chất / sản phẩm / giai đoạn / từ đồng nghĩa / nhóm cây.\n"
        "- Tránh paraphrase đơn thuần. Không tạo câu quá dài.\n"
        "- Không bịa tags. Không thêm ký tự lạ.\n"
    )

    payload = {
        "query": (norm_query or "").strip(),
        "must_tags": must_tags or [],
        "any_tags": any_tags or [],
        "answer_mode_hint": answer_mode_hint or "",
        "max_variants": max_variants,
        "output_schema": {
            "variants": [
                {"purpose": "active_ingredient|product|stage|synonym|crop_group|en|general", "q": "string"}
            ]
        },
        "notes": [
            "Nếu query có 'hoạt chất' thì bắt buộc có ít nhất 1 variant purpose=active_ingredient.",
            "Nếu có giai đoạn (trái non/ra hoa/đậu trái) thì tạo 1 variant purpose=stage.",
            "Nếu crop là bưởi/cam/quýt/chanh thì tạo 1 variant purpose=crop_group (cây có múi/citrus).",
        ],
    }

    # Dùng chat.completions như code hiện tại của bạn
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_completion_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(raw)
        variants = data.get("variants", []) or []
    except Exception:
        variants = []

    # sanitize + dedupe
    out: List[Dict[str, str]] = []
    seen = set()
    for it in variants:
        purpose = str(it.get("purpose", "general")).strip()
        q = str(it.get("q", "")).strip()
        if not q:
            continue
        q_norm = re.sub(r"\s+", " ", q).strip().lower()
        if q_norm in seen:
            continue
        if len(q_norm) > 180:
            continue
        seen.add(q_norm)
        out.append({"purpose": purpose, "q": q})
        if len(out) >= max_variants:
            break

    return out


def _purpose_weight(purpose: str) -> float:
    p = (purpose or "").strip().lower()
    # Weight theo mục tiêu (có thể chỉnh)
    if p == "active_ingredient":
        return 1.30
    if p == "stage":
        return 1.20
    if p == "crop_group":
        return 1.10
    if p == "synonym":
        return 1.00
    if p == "product":
        return 0.90
    if p == "en":
        return 0.80
    return 0.85


def _weighted_rrf_fuse(
    results_by_query: List[Dict[str, Any]],
    *,
    k: int = 60,
    top_n: int = 400,
) -> List[Dict[str, Any]]:
    """
    Weighted RRF fuse nhiều list hits (hit là dict có 'id', 'score', ...).
    Kết quả: list hits đã dedupe theo id, có thêm field:
      - hit['mq_rrf'] = điểm fused
      - hit['mq_sources'] = list purpose/query_index đã đóng góp (debug)
    """
    fused: Dict[str, float] = {}
    best: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, List[str]] = {}

    for item in results_by_query:
        weight = float(item["weight"])
        purpose = item["purpose"]
        qi = item["qi"]
        hits = item["hits"] or []

        for rank, h in enumerate(hits, start=1):
            doc_id = h.get("id")
            if not doc_id:
                continue
            rrf = weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + rrf

            # giữ bản hit tốt nhất theo score gốc (để không mất fields)
            if doc_id not in best or float(h.get("score", 0.0)) > float(best[doc_id].get("score", 0.0)):
                best[doc_id] = h

            sources.setdefault(doc_id, []).append(f"{purpose}#{qi}")

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    out: List[Dict[str, Any]] = []
    for doc_id, s in ranked:
        h = best[doc_id]
        h["mq_rrf"] = float(s)
        h["mq_sources"] = sources.get(doc_id, [])
        out.append(h)
    return out

def infer_answer_intent(q: str, found_groups: Dict[str, List[str]]):
    qn = _norm(q)

    has_product_term = bool(re.search(r"\b(thuoc|thuốc|san pham|sản phẩm)\b", qn))
    has_symptom_term = bool(re.search(r"\b(trieu chung|triệu chứng|benh|bệnh)\b", qn))

    if has_symptom_term:
        return "disease"
    if has_product_term:
        return "product"
    return "general"

def render_global_bounded(
    *,
    client,
    user_query: str,
    norm_query: str,
    analysis: Dict[str, Any],
    max_tokens: int = 700,
) -> str:
    """
    Trả lời khung nguyên tắc + checklist + hỏi tối đa 2 câu.
    KHÔNG bịa nhãn/PHI/liều.
    """
    sys = (
        "Bạn là chuyên gia BVTV/nông học.\n"
        "Trả lời theo nguyên tắc an toàn:\n"
        "- Nếu thiếu dữ liệu quyết định (PHI/nhãn/cây trồng/hoạt chất/thời điểm) thì KHÔNG kết luận khẳng định.\n"
        "- Trả lời ngắn, đúng trọng tâm.\n"
        "- Có 3 phần:\n"
        "  (1) Nguyên tắc (2-4 gạch đầu dòng)\n"
        "  (2) Checklist thông tin cần có (gạch đầu dòng)\n"
        "  (3) Hỏi lại tối đa 2 câu (nếu analysis.should_ask_back=true)\n"
    )

    # Tận dụng analysis để chắc chắn không hỏi lan man
    payload = {
        "user_query": user_query,
        "norm_query": norm_query,
        "intent_type": analysis.get("intent_type", ""),
        "required_slots": analysis.get("required_slots", []),
        "missing_slots": analysis.get("missing_slots", []),
        "ask_back_questions": analysis.get("ask_back_questions", []),
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def _is_hard_global(q: str) -> bool:
    q = (q or "").lower()
    # Các câu dạng taxonomy/liệt kê/so sánh thường cần model mạnh hơn và output dài hơn
    return any(k in q for k in [
        "bao gồm", "gồm những", "gồm các", "phân loại", "nhóm nào",
        "khác gì", "so sánh", "phân biệt", "triệu chứng", "cơ chế"
    ])

def _global_system_prompt() -> str:
    return """
Bạn là chuyên gia BVTV/nông học.
Trả lời theo kiểu giáo trình, dùng thuật ngữ phổ biến tại Việt Nam.

Quy tắc:
- Trả lời theo cấu trúc rõ ràng, có tiêu đề.
- Bắt buộc có: (1) Định nghĩa ngắn gọn, (2) Tiêu chí nhận biết/đặc điểm chính.
- Nếu câu hỏi hỏi “bao gồm/gồm những loại nào/phân loại” thì bắt buộc có mục: (3) Phân loại + (4) Ví dụ đại diện (ưu tiên nhóm/loài thường gặp trong canh tác).
- Nếu không chắc một tên loài/thuật ngữ: ghi “thường gặp” và mô tả theo nhóm, không bịa.
- Ưu tiên trả lời đúng trọng tâm, không lan man sang công dụng phủ đất/chống xói mòn nếu không liên quan câu hỏi.
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

    # KHÓA ROUTE: nếu router đã chọn RAG thì CẤM quay về GLOBAL
    route_locked = (route == "RAG")

    # -----------------------------------------------------
    # 1) NHÁNH GLOBAL NGAY TỪ ĐẦU (chỉ khi router cho phép)
    # -----------------------------------------------------
    if route == "GLOBAL":
        # hard = _is_hard_global(user_query)
        # model = "gpt-4.1" if hard else "gpt-4.1-mini"
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
            any_tags=any_tags,
            top_k=RAGConfig.multi_query_top_k
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

    # prof = analyze_hits_fused(hits)
    # strategy = decide_strategy(
    #     norm_query=norm_query,
    #     prof=prof,
    #     has_main=True,
    #     policy=policy,
    #     code_boost_direct=cfg.code_boost_direct,
    # )

    primary_doc = hits[0]

    # -----------------------------------------------------
    # 7) DIRECT DOC
    # -----------------------------------------------------
    # if strategy == "DIRECT_DOC":
    #     img_keys = extract_img_keys(primary_doc.get("answer", ""))
    #     text = format_direct_doc_answer(user_query, primary_doc)
    #     return {
    #         "text": text,
    #         "img_keys": img_keys,
    #         "route": "RAG",
    #         "norm_query": norm_query,
    #         "strategy": strategy,
    #         "profile": prof,
    #     }

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

    # final_answer = enrich_answer_if_needed(
    #     client=client,
    #     user_query=user_query,
    #     answer_text=final_answer,
    #     answer_mode=answer_mode_final,
    #     any_tags=any_tags,
    #     must_tags=must_tags,
    #     route="RAG",
    # )

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
