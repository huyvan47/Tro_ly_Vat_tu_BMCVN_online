from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score, analyze_hits_fused
from rag.strategy import decide_strategy
from rag.text_utils import extract_codes_from_query
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.answer_modes import decide_answer_policy
from rag.formatter import format_direct_doc_answer
from rag.generator import call_finetune_with_context
from rag.verbatim import verbatim_export
from rag.tag_filter import infer_filters_from_query
from rag.logger import get_logger, new_trace_id
from typing import List, Tuple
import re


logger = get_logger()

def choose_top_k(
    must_tags: List[str],
    any_tags: List[str],
    norm_query: str,
) -> int:
    """
    Quy tắc tăng top_k:
    - Nếu có entity:product => tăng mạnh (truy vấn về sản phẩm)
    - Nếu có pest:/disease: => tăng mạnh (truy vấn điều trị sâu/bệnh cần recall cao)
    - Nếu có crop: => tăng vừa (narrow theo cây trồng)
    - Nếu query có ý hỏi "thuốc gì/phun gì/trị gì" => tăng thêm
    """

    tags_all = (must_tags or []) + (any_tags or [])

    has_product = any(t == "entity:product" or t.startswith("entity:") or t.startswith("product:") for t in tags_all)
    has_pest = any(t.startswith("pest:") for t in tags_all)
    has_disease = any(t.startswith("disease:") for t in tags_all)
    has_crop = any(t.startswith("crop:") for t in tags_all)

    # Heuristic theo intent ngôn ngữ
    ask_recommend = bool(re.search(r"\b(thuoc|phun|tri|phong|xu ly|dung gi|nen dung|loai nao)\b", norm_query.lower()))

    # Tăng mạnh cho bài toán "tìm sản phẩm / tư vấn sâu bệnh"
    if has_product:
        top_k = 220

    if has_pest or has_disease:
        top_k = 300

    # Crop giúp thu hẹp, nhưng vẫn cần recall nếu kèm pest/disease
    if has_crop and not (has_pest or has_disease):
        top_k = 160

    if ask_recommend:
        top_k = int(top_k * 1.2)

    # Giới hạn trên để tránh quá tải
    top_k = min(top_k, 400)

    return top_k

def answer_with_suggestions(*, user_query, kb, client, cfg, policy):
    # 0) Route GLOBAL / RAG
    route = route_query(client, user_query)
    if route == "GLOBAL":
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            max_completion_tokens=2500,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia BVTV, giải thích khái niệm, viết tắt, định nghĩa đầy đủ, chuẩn giáo trình."
                },
                {"role": "user", "content": user_query},
            ],
        )
        return {
            "text": resp.choices[0].message.content.strip(),
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": "",
            "strategy": "GLOBAL",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # 1) Normalize query
    norm_query = normalize_query(client, user_query)

    # 2) Listing?

    must_tags, any_tags = infer_filters_from_query(norm_query)

    top_k = choose_top_k(
        must_tags=must_tags,
        any_tags=any_tags,
        norm_query=norm_query,
    )
    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)
    hits = retrieve_search(
    client=client,
    kb=kb,
    norm_query=norm_query,
    top_k=top_k,
    must_tags=must_tags,
    any_tags=any_tags,
    
)
    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "NO_HITS",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # 4) Filter by MIN_SCORE_MAIN
    for h in hits:
        h["fused_score"] = fused_score(h)
    # sort hits by fused_score desc to make profile stable
    hits = sorted(hits, key=lambda x: x["fused_score"], reverse=True)
    filtered_for_main = [h for h in hits if h["fused_score"] >= policy.min_score_main]

    # 5) Decide strategy (DIRECT_DOC / RAG_STRICT / RAG_SOFT)
    has_main = len(filtered_for_main) > 0
    prof = analyze_hits_fused(hits)
    strategy = decide_strategy(
        norm_query=norm_query,
        prof=prof,
        has_main=has_main,
        policy=policy,
        code_boost_direct=cfg.code_boost_direct,
    )

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

    # 10) Build context and call GPT answer (STRICT/SOFT)
    policy = decide_answer_policy(user_query, primary_doc)
    answer_mode = "listing" if policy.format == "listing" else policy.intent
    # === VERBATIM short-circuit ===
    if answer_mode == "verbatim":
        vb = verbatim_export(
            kb=kb,
            hits_router=hits,        # dùng hits hiện có (đã sorted/reranked), khỏi retrieve lại
        )
        return {
            "text": vb.get("text", ""),
            "img_keys": vb.get("img_keys", []),
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "VERBATIM",
            "profile": prof,
        }

    # 9) DIRECT_DOC: KB đủ mạnh -> trả trực tiếp doc (không ép QA)
    if strategy == "DIRECT_DOC":
        text = format_direct_doc_answer(user_query, primary_doc)
        return {
            "text": text,
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": strategy,
            "profile": prof,
        }

    # adaptive ctx, nhưng giới hạn theo mode
    base_ctx = choose_adaptive_max_ctx(hits)
    if strategy == "RAG_SOFT":
        max_ctx = min(RAGConfig.max_ctx_soft, base_ctx)
        rag_mode = "SOFT"
    else:
        max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)
        rag_mode = "STRICT"

    main_hits = [primary_doc]
    for h in context_candidates:
        if h is not primary_doc and len(main_hits) < max_ctx:
            main_hits.append(h)

    context = build_context_from_hits(main_hits)

    final_answer = call_finetune_with_context(
        client=client,
        user_query=user_query,
        context=context,
        answer_mode=answer_mode,
        rag_mode=rag_mode,  
    )

    return {
        "text": final_answer,
        "route": "RAG",
        "norm_query": norm_query,
        "strategy": strategy,
        "profile": prof,
    }