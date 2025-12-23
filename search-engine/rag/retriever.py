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

    def doc_has_tags(i: int, must_local, any_local) -> bool:
        """
        Hard filter using TAGS_V2.
        - must_local: AND
        - any_local : OR
        If TAGS_V2 missing, always True (old behavior).
        """
        if TAGS_V2 is None:
            return True

        if not must_local and not any_local:
            return True

        tagset = _parse_tags_any_format(TAGS_V2[i])

        # If require must tags but doc has no tagset => reject
        if not tagset:
            return False if must_local else True

        if must_local and not all(t in tagset for t in must_local):
            return False
        if any_local and not any(t in tagset for t in any_local):
            return False
        return True

    def pick_indices(must_local, any_local):
        picked_local = []
        for j in idx_sorted:
            j = int(j)
            if doc_has_tags(j, must_local, any_local):
                picked_local.append(j)
                if len(picked_local) >= top_k:
                    break
        return picked_local

    # --- Strict first ---
    picked = pick_indices(must_tags, any_tags)

    # --- Fallback 1: drop any_tags only ---
    if len(picked) < top_k and any_tags:
        picked = pick_indices(must_tags, [])

    # --- Fallback 2: drop must_tags (full recall) ---
    if len(picked) < top_k and must_tags:
        picked = pick_indices([], [])

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
