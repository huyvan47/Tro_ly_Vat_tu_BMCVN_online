import json
import numpy as np
from rag.config import RAGConfig
from rag.reranker import llm_rerank


def embed_query(client, text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def _parse_tags_any_format(x):
    """
    Accept tags stored as:
    - None
    - "a|b|c"
    - '["a","b"]' (JSON array string)
    - python list/np array
    Return: set(str)
    """
    if x is None:
        return set()

    if isinstance(x, (list, tuple, set)):
        return set(str(t).strip() for t in x if str(t).strip())

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return set()

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return set(str(t).strip() for t in arr if str(t).strip())
        except Exception:
            pass

    if "|" in s:
        return set(p.strip() for p in s.split("|") if p.strip())

    return {s}


def search(client, kb, norm_query: str, top_k: int, must_tags=None, any_tags=None):
    """
    must_tags: list[str] -> AND condition (must include all)
    any_tags : list[str] -> OR condition (must include at least one)
    If TAGS_V2 is missing in KB, filtering is skipped (backward compatible).
    """
    must_tags = list(must_tags or [])
    any_tags = list(any_tags or [])

    # Backward compatibility:
    # old: (EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS)
    # new: (EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS, TAGS_V2, ENTITY_TYPE)
    if len(kb) >= 9:
        EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS, TAGS_V2, ENTITY_TYPE = kb
    else:
        EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS = kb
        TAGS_V2 = None
        ENTITY_TYPE = None

    # --- Query embedding ---
    q = embed_query(client, norm_query)

    # --- Similarity ---
    embs = np.array(EMBS, dtype=np.float32)
    sims = embs @ q
    idx_sorted = np.argsort(-sims)

    debug = True
    debug_limit = 120  # tăng nếu top_k lớn

    def explain_doc_tags(i: int, must_local, any_local):
        if TAGS_V2 is None:
            # không filter => không bonus
            return True, "PASS: TAGS_V2 is None -> skip filtering", set(), False, 0

        tagset = _parse_tags_any_format(TAGS_V2[i])

        # Không có must/any => full recall => PASS nhưng KHÔNG bonus
        if not must_local and not any_local:
            return True, "PASS: no must/any provided", tagset, False, 0

        # must
        missing_must = [t for t in must_local if t not in tagset]
        if missing_must:
            return False, f"FAIL: missing must_tags={missing_must}", tagset, False, 0

        # any
        if any_local:
            hit_any = [t for t in any_local if t in tagset]
            if not hit_any:
                return False, f"FAIL: none of any_tags matched (need one of {any_local})", tagset, False, 0
            # pass do match any => được bonus
            return True, f"PASS: matched any_tags={hit_any}", tagset, True, len(hit_any)

        # pass do match must => được bonus
        return True, "PASS: matched all must_tags", tagset, True, len(must_local)

    def pick_indices(must_local, any_local, stage_name: str):
        picked_local = []

        # if debug:
        #     print(f"\n=== PICK STAGE: {stage_name} ===")
        #     print("must_local:", must_local)
        #     print("any_local :", any_local)
        #     print("top_k     :", top_k)
        #     if TAGS_V2 is None:
        #         print("NOTE: TAGS_V2 is None => tag filtering DISABLED\n")

        # Nếu không có tag filter, đừng bonus để tránh "cộng bừa"
        has_tag_filter = bool(must_local or any_local)

        # Nếu KB nhỏ thì cứ xét all; nếu KB lớn, vẫn có thể dùng WINDOW nhưng nên lớn hơn nhiều
        # Khuyến nghị: ưu tiên xét ALL các doc ok=True (lọc theo tag) thay vì cắt theo sim trước.
        # Ta sẽ:
        # 1) duyệt all idx_sorted nhưng chỉ collect doc ok=True (và stop sớm khi đủ "pool")
        # 2) rank pool theo (ok, matches, sim)
        #
        # pool_size nên lớn hơn top_k nhiều để ổn định.
        pool_size = max(top_k * 200, 3000)  # tăng mạnh so với 40 để tag-match có cửa

        scored = []
        inspected = 0
        collected_ok = 0

        # Duyệt theo sim giảm dần để lấy pool các doc "ok"
        for j in idx_sorted:
            j = int(j)
            ok, reason, tagset, bonus_ok, num_matches = explain_doc_tags(j, must_local, any_local)
            sim = float(sims[j])

            if debug and inspected < debug_limit:
                qtxt = str(QUESTIONS[j]) if QUESTIONS is not None else ""
                et   = str(ENTITY_TYPE[j]) if ENTITY_TYPE is not None else ""
                tv2  = str(TAGS_V2[j]) if TAGS_V2 is not None else ""

                # Key dùng để sort (ưu tiên match, rồi sim)
                # - Nếu có tag filter: ưu tiên num_matches
                # - Nếu không có tag filter: num_matches=0 cho tất cả
                key = (
                    1 if ok else 0,
                    int(num_matches) if has_tag_filter else 0,
                    sim
                )

                print(
                    f"[{stage_name}] cand#{inspected:03d} idx={j} "
                    f"sim={sim:.4f} ok={ok} matches={int(num_matches)} key={key}"
                )
                print("  entity_type:", et)
                print("  question   :", qtxt[:120])
                print("  tags_v2_raw:", tv2[:200])
                if tagset:
                    ts = sorted(tagset)
                    print("  tagset     :", ts[:40], ("...(+%d)" % (len(ts)-40) if len(ts) > 40 else ""))
                print("  reason     :", reason)
                print("-" * 60)

            inspected += 1

            if ok:
                # Lưu (key, idx) để sort lại
                key = (
                    1,
                    int(num_matches) if has_tag_filter else 0,
                    sim
                )
                scored.append((key, j, sim, int(num_matches), reason, tagset))
                collected_ok += 1

                # đủ pool thì dừng (để tiết kiệm)
                if collected_ok >= pool_size:
                    break

        # Sort: match trước, rồi sim
        scored.sort(key=lambda x: x[0], reverse=True)

        # Lấy top_k
        for key, j, sim, nm, reason, tagset in scored:
            picked_local.append(int(j))
            if len(picked_local) >= top_k:
                break

        if debug:
            print(f"=== PICKED {len(picked_local)}/{top_k} in stage {stage_name} ===")
            for r, idx in enumerate(picked_local[:min(len(picked_local), 30)], 1):
                tv2 = str(TAGS_V2[idx]) if TAGS_V2 is not None else ""
                # in cả sim + matches để nhìn thấy lý do đứng top
                ok, reason, tagset, bonus_ok, num_matches = explain_doc_tags(idx, must_local, any_local)
                print(f"  #{r:02d} idx={idx} sim={float(sims[idx]):.4f} matches={int(num_matches)} id={IDS[idx] if IDS is not None else ''}")
                print(f"      Q: {str(QUESTIONS[idx])[:140] if QUESTIONS is not None else ''}")
                print(f"      tags_v2: {tv2[:200]}")
            print()

        return picked_local


    final_stage = None
    # --- Strict first ---
    picked = pick_indices(must_tags, any_tags, "STRICT")
    final_stage = "STRICT"
    # --- Fallback 1: drop any_tags only ---
    if len(picked) < top_k and any_tags:
        picked = pick_indices(must_tags, [], "FALLBACK1_DROP_ANY")
        final_stage = "FALLBACK1_DROP_ANY"
    # --- Fallback 2: drop must_tags (full recall) ---
    if len(picked) < top_k and must_tags:
        picked = pick_indices([], [], "FALLBACK2_DROP_MUST_FULL_RECALL")
        final_stage = "FALLBACK2_DROP_MUST_FULL_RECALL"
    if debug:
        print("\n=== FINAL PICK STAGE ===")
        print("final_stage :", final_stage)
        print("picked_count:", len(picked))
        print("top_k       :", top_k)
        print("========================\n")
    # --- Build results ---
    results = []
    for i in picked:
        item = {
            "id": str(IDS[i]) if IDS is not None else "",
            "question": str(QUESTIONS[i]) if QUESTIONS is not None else "",
            "alt_question": str(ALT_QUESTIONS[i]) if ALT_QUESTIONS is not None else "",
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        }

        if CATEGORY is not None:
            item["category"] = str(CATEGORY[i])

        if ENTITY_TYPE is not None:
            item["entity_type"] = str(ENTITY_TYPE[i])

        # Prefer tags_v2 for debug/metadata
        if TAGS_V2 is not None:
            item["tags_v2"] = str(TAGS_V2[i])
        elif TAGS is not None:
            item["tags"] = str(TAGS[i])

        results.append(item)

    # --- Optional rerank ---
    # NOTE: Nếu query dạng "liệt kê theo hoạt chất", rerank LLM có thể làm giảm độ đầy đủ.
    # Bạn có thể cân nhắc disable rerank khi must_tags có "chemical:*".
    if RAGConfig.use_llm_rerank and len(results) > 1:
        results = llm_rerank(client, norm_query, results, RAGConfig.top_k_rerank)

    return results
