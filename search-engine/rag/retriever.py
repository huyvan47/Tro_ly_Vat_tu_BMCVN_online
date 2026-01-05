import json
import numpy as np
from rag.config import RAGConfig
from rag.reranker import llm_rerank
from rag.debug_log import debug_log


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
    must_tags = list(must_tags or [])
    any_tags  = list(any_tags or [])

    # NEW: filter gốc của query
    base_has_filter = bool(must_tags or any_tags)

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
        has_tag_filter = bool(must_local or any_local)

        pool_size = max(top_k * 200, 3000)

        scored = []
        inspected = 0
        collected_ok = 0

        for j in idx_sorted:
            j = int(j)
            ok, reason, tagset, bonus_ok, num_matches = explain_doc_tags(j, must_local, any_local)
            sim = float(sims[j])

            inspected += 1

            if ok:
                # NEW: nếu query có filter thì cấm doc matches=0
                if base_has_filter and int(num_matches) <= 0:
                    continue

                key = (1, int(num_matches), sim)
                scored.append((key, j, sim, int(num_matches), reason, tagset))
                collected_ok += 1

                if collected_ok >= pool_size:
                    break

        scored.sort(key=lambda x: x[0], reverse=True)

        for key, j, sim, nm, reason, tagset in scored:
            picked_local.append(int(j))
            if len(picked_local) >= top_k:
                break

        if debug:
            debug_log(f"=== PICKED {len(picked_local)}/{top_k} in stage {stage_name} ===")
            for r, idx in enumerate(picked_local[:min(len(picked_local), 30)], 1):
                tv2 = str(TAGS_V2[idx]) if TAGS_V2 is not None else ""

                # NEW: in matches theo filter gốc của query để log không gây hiểu nhầm
                ok0, reason0, tagset0, bonus_ok0, num_matches0 = explain_doc_tags(idx, must_tags, any_tags)

                debug_log(
                    f"  #{r:02d} idx={idx} sim={float(sims[idx]):.4f} matches={int(num_matches0)} id={IDS[idx] if IDS is not None else ''}",
                    f"      Q: {str(QUESTIONS[idx])[:140] if QUESTIONS is not None else ''}",
                    f"      tags_v2: {tv2[:200]}"
                )
            debug_log("")

        return picked_local

    def merge_fill(primary, secondary, top_k):
        seen = set(primary)
        out = list(primary)
        for j in secondary:
            if j in seen:
                continue
            out.append(j)
            seen.add(j)
            if len(out) >= top_k:
                break
        return out

    picked_strict = pick_indices(must_tags, any_tags, "STRICT")
    picked = picked_strict
    final_stage = "STRICT"

    final_stage = None
    # --- Strict first ---
    picked = pick_indices(must_tags, any_tags, "STRICT")
    final_stage = "STRICT"
    if len(picked) < top_k and any_tags:
        picked_fb1 = pick_indices(must_tags, [], "FALLBACK1_DROP_ANY")
        picked = merge_fill(picked, picked_fb1, top_k)
        final_stage = "STRICT+FALLBACK1"

    if len(picked) < top_k and must_tags:
        picked_fb2 = pick_indices([], [], "FALLBACK2_DROP_MUST_FULL_RECALL")
        picked = merge_fill(picked, picked_fb2, top_k)
        final_stage = "STRICT+FALLBACK1+FALLBACK2"

    if debug:
        debug_log(
            "=== FINAL PICK STAGE ===",
            f"final_stage : {final_stage}",
            f"picked_count: {len(picked)}",
            f"top_k       : {top_k}",
            "========================"
        )
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
