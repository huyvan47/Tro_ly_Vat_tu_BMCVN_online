import os
import re
import numpy as np
from openai import OpenAI

# =========================
# CONFIG
# =========================
NPZ_PATH = "data-vat-tu-full-merge-overlap.npz"

EMBED_MODEL = "text-embedding-3-small"

# Pagination: tổng số ký tự chunk_text đưa vào 1 lần gọi GPT
MAX_SOURCE_CHARS_PER_CALL = 12000

# Retrieval
TOPK_STAGE1 = 8  # lấy vài ứng viên để vote parent

# =========================
# LOAD NPZ
# =========================
data = np.load(NPZ_PATH, allow_pickle=True)

embs = data["embeddings"].astype(np.float32)  # assumed normalized
ids = data["ids"].astype(object)

# NOTE: trong NPZ script trước của bạn, "answers" chính là df["answer"].
# Nếu bạn đã xuất chunk vào cột "answer" và nó chính là chunk_text, thì ok.
chunk_texts = data["answers"].astype(object)  # treat as chunk_text

# metadata (không bắt buộc cho verbatim)
questions = data.get("questions", None)
alts = data.get("alt_questions", None)
categories = data.get("category", None)
tags = data.get("tags", None)

client = OpenAI(api_key="...")
  # set env var recommended


# =========================
# HELPERS
# =========================
def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def parse_parent_and_index(chunk_id: str):
    """
    Expect: <parent>_chunk_<NN>
    Returns: (parent_id, chunk_index) or (chunk_id, 0) if not match.
    """
    s = str(chunk_id)
    m = re.match(r"^(.*?)_chunk_(\d+)$", s)
    if not m:
        return s, 0
    return m.group(1), int(m.group(2))

def embed_query(text: str) -> np.ndarray:
    text = normalize_query(text)
    # dùng format chuẩn giống bạn đang làm (Q/ALT/A)
    # Ở query chỉ cần Q là đủ
    resp = client.embeddings.create(model=EMBED_MODEL, input=[f"Q: {text}"])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int):
    scores = matrix @ query_vec
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]

def choose_parent_by_vote(candidate_ids):
    """
    Vote parent among candidate chunk ids (top-k)
    """
    parents = []
    for cid in candidate_ids:
        p, _ = parse_parent_and_index(cid)
        parents.append(p)
    # majority vote
    from collections import Counter
    cnt = Counter(parents)
    parent_id, _ = cnt.most_common(1)[0]
    return parent_id

def fetch_all_chunks_by_parent(parent_id: str):
    prefix = str(parent_id) + "_chunk_"
    items = []
    for i, cid in enumerate(ids):
        s = str(cid)
        if s.startswith(prefix):
            _, cidx = parse_parent_and_index(s)
            items.append((cidx, s, str(chunk_texts[i])))
    items.sort(key=lambda x: x[0])
    return items  # list[(chunk_index, chunk_id, chunk_text)]

def paginate_chunks(items, max_chars=MAX_SOURCE_CHARS_PER_CALL):
    pages = []
    cur = []
    cur_len = 0
    for cidx, cid, txt in items:
        block_len = len(cid) + len(txt) + 20
        if cur and cur_len + block_len > max_chars:
            pages.append(cur)
            cur = []
            cur_len = 0
        cur.append((cidx, cid, txt))
        cur_len += block_len
    if cur:
        pages.append(cur)
    return pages

def build_verbatim_prompt(user_query: str, page_items):
    """
    page_items: list[(chunk_index, chunk_id, chunk_text)]
    """
    parts = []
    for cidx, cid, txt in page_items:
        parts.append(f"[CHUNK {cid}]\n{txt}\n")
    source = "\n".join(parts).strip()

    prompt = f"""Bạn đang ở chế độ TRÍCH NGUYÊN VĂN (verbatim).
Quy tắc bắt buộc:
1) Chỉ được phép xuất ra các đoạn văn bản xuất hiện nguyên văn trong phần NGUỒN bên dưới.
2) Không được diễn giải, không tóm tắt, không thêm bớt chữ, không sửa chính tả, không thay đổi dấu câu.
3) Giữ nguyên thứ tự như trong NGUỒN.
4) Nếu trong NGUỒN không có thông tin để trả lời, hãy ghi đúng câu: "KHÔNG TÌM THẤY TRONG NGUỒN".

Yêu cầu người dùng: {user_query}

NGUỒN:
{source}

Hãy xuất nguyên văn toàn bộ nội dung trong NGUỒN theo đúng thứ tự, không thêm lời dẫn.
"""
    return prompt, source

def validate_verbatim(output_text: str, source_text: str) -> bool:
    """
    Validate output is verbatim: every non-empty line must appear in source_text.
    This is strict enough to catch added words.
    """
    out_lines = [ln.strip() for ln in output_text.splitlines() if ln.strip()]
    if not out_lines:
        return False
    return all(ln in source_text for ln in out_lines)

def call_gpt_verbatim(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # bạn có thể đổi model
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# =========================
# MAIN FLOW
# =========================
def verbatim_export(user_query: str, use_gpt: bool = True):
    """
    use_gpt=True: GPT chỉ "in" lại theo prompt verbatim.
    use_gpt=False: hệ thống tự in chunk_text (an toàn tuyệt đối cho 100% chữ).
    """
    qvec = embed_query(user_query)
    idx, scores = cosine_topk(qvec, embs, TOPK_STAGE1)

    cand_ids = [str(ids[i]) for i in idx]
    parent_id = choose_parent_by_vote(cand_ids)

    all_chunks = fetch_all_chunks_by_parent(parent_id)
    if not all_chunks:
        return parent_id, ["KHÔNG TÌM THẤY TRONG NGUỒN"]

    pages = paginate_chunks(all_chunks)

    outputs = []
    for pageno, page_items in enumerate(pages, start=1):
        if not use_gpt:
            # 100% chắc: in trực tiếp
            text = "\n\n".join([txt for _, _, txt in page_items]).strip()
            outputs.append(f"=== PART {pageno}/{len(pages)} | PARENT {parent_id} ===\n{text}")
            continue

        prompt, source_text = build_verbatim_prompt(user_query, page_items)
        out = call_gpt_verbatim(prompt).strip()

        # Validate: nếu GPT thêm chữ -> fallback in trực tiếp
        if not validate_verbatim(out, source_text):
            fallback = "\n\n".join([txt for _, _, txt in page_items]).strip()
            out = fallback

        outputs.append(f"=== PART {pageno}/{len(pages)} | PARENT {parent_id} ===\n{out}")

    return parent_id, outputs


if __name__ == "__main__":
    q = "các chai không đóng được Gardona"
    parent, parts = verbatim_export(q, use_gpt=True)
    print("\n\n".join(parts))
