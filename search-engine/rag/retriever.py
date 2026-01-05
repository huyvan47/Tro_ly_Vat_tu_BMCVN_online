import re
import json
import numpy as np
from rag.config import RAGConfig
from rag.debug_log import debug_log
from rag.logger import get_logger, new_trace_id

logger = get_logger()


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
    - '[""a"", ""b""]' (CSV-escaped quotes)
    - "a,b,c" (comma separated)
    - python list/tuple/set/np array
    Return: set(str)
    """
    if x is None:
        return set()

    if isinstance(x, (list, tuple, set)):
        return set(str(t).strip() for t in x if str(t).strip())

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return set()

    # Remove wrapping quotes if the whole cell is quoted
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    # If looks like an array, try JSON (with fix for CSV-escaped quotes)
    if s.startswith("[") and s.endswith("]"):
        s_json = s
        if '""' in s_json:
            s_json = s_json.replace('""', '"')

        try:
            arr = json.loads(s_json)
            if isinstance(arr, list):
                return set(str(t).strip() for t in arr if str(t).strip())
        except Exception:
            pass

        tokens = re.findall(r'["\']([^"\']+)["\']', s)
        if tokens:
            return set(t.strip() for t in tokens if t.strip())

        inner = s[1:-1].strip()
        if inner:
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return set(p for p in parts if p)

        return set()

    # Pipe format
    if "|" in s:
        return set(p.strip() for p in s.split("|") if p.strip())

    # Comma format
    if "," in s:
        parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return set(parts)

    return {s}


def search(client, kb, norm_query: str, top_k: int, must_tags=None, any_tags=None):
    """
    must_tags: list[str] -> AND condition (must include all)
    any_tags : list[str] -> OR condition (must include at least one)
    If TAGS_V2 is missing in KB, filtering is skipped (backward compatible).

    Output item fields (added):
    - stage: "STRICT" or "FALLBACK1_DROP_ANY" or "FALLBACK2_DROP_MUST_FULL_RECALL"
    - match_count: số tag match (any hoặc must, tùy stage)
    """
    must_tags = list(must_tags or [])
    any_tags = list(any_tags or [])

    trace_id = new_trace_id()
    logger.debug(
        f"Tag filter: must={must_tags}, any={any_tags}",
        extra={"trace_id": trace_id},
    )

    # Backward compatibility:
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
    debug_limit = 120  # log candidates

    # Track per-index info for later scoring / debug
    match_count_by_idx = {}
    stage_by_idx = {}
    reason_by_idx = {}

    def explain_doc_tags(i: int, must_local, any_local):
        """
        Return:
          ok: bool
          reason: str
          tagset: set
          bonus_ok: bool (has_tag_filter and doc matched filter)
          num_matches: int (#hit any OR #must matched count)
        """
        if TAGS_V2 is None:
            # no tags -> no filtering
            return True, "PASS: TAGS_V2 is None -> skip filtering", set(), False, 0

        tagset = _parse_tags_any_format(TAGS_V2[i])

        # No filter => full recall (but no match bonus)
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
            return True, f"PASS: matched any_tags={hit_any}", tagset, True, len(hit_any)

        # only must
        return True, "PASS: matched all must_tags", tagset, True, len(must_local)

    def pick_indices(must_local, any_local, stage_name: str):
        picked_local = []
        has_tag_filter = bool(must_local or any_local)

        # Pool size: đủ lớn để có nhiều doc hợp lệ, nhưng không quá lớn gây chậm.
        # top_k=300 -> pool khoảng 2400-3600 là hợp lý.
        pool_size = max(top_k * 10, 2000)
        pool_size = min(pool_size, 20000)  # hard cap

        scored = []
        inspected = 0
        collected_ok = 0

        for j in idx_sorted:
            j = int(j)
            ok, reason, tagset, bonus_ok, num_matches = explain_doc_tags(j, must_local, any_local)
            sim = float(sims[j])

            if debug and inspected < debug_limit:
                inspected += 1

            if ok:
                # key ưu tiên:
                # - Nếu có tag_filter: ưu tiên match_count rồi sim
                # - Nếu không: chỉ sim
                key = (
                    1,
                    int(num_matches) if has_tag_filter else 0,
                    sim,
                )
                scored.append((key, j, sim, int(num_matches), reason))
                collected_ok += 1

                # record
                match_count_by_idx[j] = int(num_matches) if has_tag_filter else 0
                stage_by_idx[j] = stage_name
                reason_by_idx[j] = reason

                if collected_ok >= pool_size:
                    break

        scored.sort(key=lambda x: x[0], reverse=True)

        for _, j, _, _, _ in scored:
            picked_local.append(int(j))
            if len(picked_local) >= top_k:
                break

        if debug:
            debug_log(f"=== PICKED {len(picked_local)}/{top_k} in stage {stage_name} ===")
            for r, idx in enumerate(picked_local[: min(len(picked_local), 30)], 1):
                tv2 = str(TAGS_V2[idx]) if TAGS_V2 is not None else ""
                nm = int(match_count_by_idx.get(idx, 0))
                debug_log(
                    f"  #{r:02d} idx={idx} sim={float(sims[idx]):.4f} matches={nm} id={IDS[idx] if IDS is not None else ''}",
                    f"      Q: {str(QUESTIONS[idx])[:140] if QUESTIONS is not None else ''}",
                    f"      tags_v2: {tv2[:200]}",
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

    # --- Strict first ---
    picked = pick_indices(must_tags, any_tags, "STRICT")
    final_stage = "STRICT"

    # Fallback 1: drop ANY (keep MUST), only if ANY existed and strict not enough
    if len(picked) < top_k and any_tags:
        picked_fb1 = pick_indices(must_tags, [], "FALLBACK1_DROP_ANY")
        picked = merge_fill(picked, picked_fb1, top_k)
        final_stage = "STRICT+FALLBACK1"

    # Fallback 2: drop MUST too (full recall), only if strict still not enough
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
            "========================",
        )

    # --- Build results ---
    results = []
    for i in picked:
        i = int(i)
        base_sim = float(sims[i])

        stage = stage_by_idx.get(i, "STRICT")
        nm = int(match_count_by_idx.get(i, 0))

        # ---- Tag-match bonus + stage boost ----
        # Mục tiêu: giữ tài liệu STRICT (match tag) không bị fallback similarity thuần đẩy xuống.
        #
        # - bonus theo match_count: +0.02 mỗi match, cap 0.08
        # - stage boost: STRICT +0.05, fallback +0.00
        #
        bonus = min(0.08, 0.02 * nm)
        if stage == "STRICT":
            bonus += 0.05

        score = base_sim + bonus

        item = {
            "id": str(IDS[i]) if IDS is not None else "",
            "question": str(QUESTIONS[i]) if QUESTIONS is not None else "",
            "alt_question": str(ALT_QUESTIONS[i]) if ALT_QUESTIONS is not None else "",
            "answer": str(ANSWERS[i]),
            "score": float(score),
            "raw_sim": float(base_sim),
            "stage": stage,
            "match_count": nm,
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

    return results
