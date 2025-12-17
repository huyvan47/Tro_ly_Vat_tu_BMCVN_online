from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query, extract_img_keys
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score, analyze_hits_fused, analyze_hits
from rag.strategy import decide_strategy
from rag.text_utils import extract_codes_from_query
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.answer_modes import detect_answer_mode
from rag.formatter import format_direct_doc_answer
from rag.generator import call_finetune_with_context
from rag.verbatim import verbatim_export

def answer_with_suggestions(*, user_query, kb, client, cfg, policy, logger=None):
    # 0) Route GLOBAL / RAG
    route = route_query(client, user_query)
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
    is_list = is_listing_query(norm_query)

    top_k = 50 if is_list else 40   # giống “tinh thần” bản cũ
    hits = retrieve_search(client=client, kb=kb, norm_query=norm_query, top_k=top_k)
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

    if not filtered_for_main:
        suggestions_text = "\n".join(
            f"- {h['question']} (score={h['score']:.2f})"
            for h in hits[:10]
        )
        return {
            "text": "Không có tài liệu đủ độ tương đồng.\n\nGợi ý:\n" + suggestions_text,
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "LOW_SIM",
            "profile": analyze_hits_fused(hits),
        }
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
        if h["question"] not in used_q
        and h["fused_score"] >= policy.min_suggest_score
    ][:policy.max_suggest]
    suggestions_text = "\n".join(f"- {h['question']}" for h in suggest) if suggest else "- (không có)"

    # 10) Build context and call GPT answer (STRICT/SOFT)
    answer_mode = detect_answer_mode(user_query, primary_doc, is_list)
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
        img_keys = extract_img_keys(primary_doc.get("answer", ""))
        text = format_direct_doc_answer(user_query, primary_doc, suggestions_text)
        return {
            "text": text,
            "img_keys": img_keys,
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": strategy,
            "profile": prof,
        }

    # adaptive ctx, nhưng giới hạn theo mode
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
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
        suggestions_text=suggestions_text,
        answer_mode=answer_mode,
        rag_mode=rag_mode,  
    )

    img_keys = extract_img_keys(primary_doc.get("answer", ""))
    return {
        "text": final_answer,
        "img_keys": img_keys,
        "route": "RAG",
        "norm_query": norm_query,
        "strategy": strategy,
        "profile": prof,
    }