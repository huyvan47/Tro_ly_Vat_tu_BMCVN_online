import re
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

def parse_doc_tags(doc):
    """
    Normalize tags from document into a SET[str].

    Supported formats:
    - doc["tags_v2"] as list[str]
    - doc["tags_v2"] as comma-separated string
    - doc["tags_v2"] as dict (keys are tags)
    - fallback: try doc["tags"]

    Output:
    - set of lowercase tag strings
    """

    tags = set()

    if not doc:
        return tags

    raw = None

    if isinstance(doc, dict):
        if "tags_v2" in doc:
            raw = doc.get("tags_v2")
        elif "tags" in doc:
            raw = doc.get("tags")

    # Case 1: list of tags
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, str):
                tags.add(t.strip().lower())

    # Case 2: dict of tags
    elif isinstance(raw, dict):
        for k in raw.keys():
            if isinstance(k, str):
                tags.add(k.strip().lower())

    # Case 3: string "a|b|c" or "a,b,c"
    elif isinstance(raw, str):
        s = raw.replace("|", ",")
        for part in s.split(","):
            t = part.strip().lower()
            if t:
                tags.add(t)

    return tags

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

    def _parse_tags_any_format(x):
        if x is None:
            return set()

        if isinstance(x, (list, tuple, set)):
            return {str(t).strip().lower() for t in x if str(t).strip()}

        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return set()

        # unwrap if the whole cell is quoted
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()

        # JSON-like list
        if s.startswith("[") and s.endswith("]"):
            s_json = s.replace('""', '"')
            try:
                arr = json.loads(s_json)
                if isinstance(arr, list):
                    return {str(t).strip().lower() for t in arr if str(t).strip()}
            except Exception:
                pass

            tokens = re.findall(r'["\']([^"\']+)["\']', s)
            if tokens:
                return {t.strip().lower() for t in tokens if t.strip()}

            inner = s[1:-1].strip()
            if inner:
                parts = [p.strip().strip('"').strip("'").lower() for p in inner.split(",")]
                return {p for p in parts if p}

            return set()

        # pipe
        if "|" in s:
            return {p.strip().lower() for p in s.split("|") if p.strip()}

        # comma
        if "," in s:
            parts = [p.strip().strip('"').strip("'").lower() for p in s.split(",")]
            return {p for p in parts if p}

        return {s.lower()}


    def explain_doc_tags(doc_tags, must_tags, any_tags):
        # No filters => PASS, no bonus notion here (bonus handled by compute_tag_score)
        if not must_tags and not any_tags:
            return True, "PASS: no must/any provided"

        missing_must = [t for t in must_tags if t not in doc_tags]
        if missing_must:
            return False, f"FAIL: missing must_tags={missing_must}"

        if any_tags:
            hit_any = [t for t in any_tags if t in doc_tags]
            if not hit_any:
                return False, f"FAIL: none of any_tags matched (need one of {any_tags})"
            return True, f"PASS: matched any_tags={hit_any}"

        return True, "PASS: matched all must_tags"


    def compute_tag_score(doc_tags, must_tags, any_tags):
        score = 0
        for t in must_tags:
            if t in doc_tags:
                score += 3
        for t in any_tags:
            if t in doc_tags:
                score += 1
        return score

    def log_pick(stage_name, picked_rows, IDS, QUESTIONS, TAGS_V2):
        debug_log("=== PICKED {} in stage {} ===".format(len(picked_rows), stage_name))

        for rank, row in enumerate(picked_rows, 1):
            _, i, sim, tag_score, reason = row

            did = str(IDS[i]) if IDS is not None else ""
            q = str(QUESTIONS[i]) if QUESTIONS is not None else ""
            tags = str(TAGS_V2[i]) if TAGS_V2 is not None else ""

            debug_log(
                "#{:02d} idx={} sim={:.4f} tag_score={} id={}".format(
                    rank, i, float(sim), int(tag_score), did
                ),
                "    Q: " + q,
                "    reason: " + str(reason),
                "    tags_v2: " + tags,
            )


    def pick_indices(stage_name, must_local, any_local, top_k):
        scored = []

        for idx in idx_sorted:
            i = int(idx)  # ensure python int index

            sim = float(sims[i])

            if TAGS_V2 is None:
                doc_tags = set()
            else:
                doc_tags = _parse_tags_any_format(TAGS_V2[i])

            if stage_name == "FALLBACK1_DROP_ANY":
                ok, reason = explain_doc_tags(doc_tags, must_local, [])
                # ranking still uses original any_tags to reward relevant docs
                tag_score = compute_tag_score(doc_tags, must_local, any_tags)
            else:
                ok, reason = explain_doc_tags(doc_tags, must_local, any_local)
                tag_score = compute_tag_score(doc_tags, must_local, any_local)

            key = (1 if ok else 0, int(tag_score), sim)
            scored.append((key, i, sim, int(tag_score), reason))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked_rows = scored[:top_k]

        if debug:
            log_pick(stage_name, picked_rows, IDS, QUESTIONS, TAGS_V2)

        return [i for _, i, _, _, _ in picked_rows]


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
    picked = pick_indices("STRICT", must_tags, any_tags, top_k)
    final_stage = "STRICT"

    # Fallback 1: drop any (do not fail on any)
    if len(picked) < top_k and any_tags:
        picked_fb1 = pick_indices("FALLBACK1_DROP_ANY", must_tags, [], top_k)
        before = len(picked)
        picked = merge_fill(picked, picked_fb1, top_k)
        if len(picked) > before:
            final_stage = "STRICT+FALLBACK1"

    # Fallback 2: full recall by sim
    if len(picked) < top_k:
        picked_fb2 = pick_indices("FALLBACK2_SIM_FULL_RECALL", [], [], top_k)
        before = len(picked)
        picked = merge_fill(picked, picked_fb2, top_k)
        if len(picked) > before:
            final_stage = (final_stage + "+FALLBACK2") if final_stage else "FALLBACK2"

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
