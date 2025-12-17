import re
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from rag.config import RAGConfig

def extract_img_keys(text: str) -> List[str]:
    return re.findall(r"\(IMG_KEY:\s*([^)]+)\)", str(text))

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

def fetch_all_chunks_by_parent(kb, parent_id: str):
    EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS = kb

    prefix = str(parent_id) + "_chunk_"
    items = []
    for i, cid in enumerate(IDS):
        s = str(cid)
        if s.startswith(prefix):
            _, cidx = parse_parent_and_index(s)
            items.append((cidx, s, str(ANSWERS[i])))
    items.sort(key=lambda x: x[0])
    return items


def paginate_chunks(items: List[Tuple[int, str, str]], max_chars: int = RAGConfig.max_source_chars_per_call):
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

def verbatim_export(kb, hits_router: List[Dict[str, Any]]) -> Dict[str, Any]:
    parent_id = choose_parent_by_weighted_vote(hits_router[:RAGConfig.topk_router])

    # If the chosen parent has no chunks, fallback to best single doc (atomic)
    chunks = fetch_all_chunks_by_parent(kb, parent_id)
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
