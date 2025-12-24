from rag.config import RAGConfig
import json

def _parse_bool(v):
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False

def llm_rerank(client, norm_query: str, results: list, top_k_rerank: int = RAGConfig.top_k_rerank):
    if top_k_rerank is None:
        top_k_rerank = RAGConfig.top_k_rerank

    if not results or len(results) == 1:
        return results

    candidates = results[:top_k_rerank]

    doc_texts = []
    for i, h in enumerate(candidates):
        ans = str(h.get("answer", ""))
        if len(ans) > RAGConfig.rerank_snippet_chars:
            ans = ans[:RAGConfig.rerank_snippet_chars] + " ..."
        doc_texts.append(
            f"[DOC {i}]\n"
            f"QUESTION: {h.get('question','')}\n"
            f"ALT_QUESTION: {h.get('alt_question','')}\n"
            f"ANSWER_SNIPPET:\n{ans}"
        )

    docs_block = "\n\n------------------------\n\n".join(doc_texts)
    if RAGConfig.debug_rerank:
        print("docs_block:\n", docs_block)

    system_prompt = (
        "Bạn là LLM dùng để RERANK tài liệu cho hệ thống trợ lý nội bộ BMCVN. "
        "Bạn CHỈ được trả về JSON THUẦN (không markdown, không code block)."
    )

    user_prompt = f"""
CÂU HỎI:
\"\"\"{norm_query}\"\"\"

CÁC TÀI LIỆU ỨNG VIÊN:
{docs_block}

YÊU CẦU:
- Với mỗi DOC:
  (1) Chấm điểm liên quan 0–1
  (2) include_in_context: true/false (nên đưa vào NGỮ CẢNH trả lời hay không)

- Trả về DUY NHẤT JSON dạng:
[
  {{ "doc_index": 0, "score": 0.92, "include_in_context": true }},
  {{ "doc_index": 1, "score": 0.30, "include_in_context": false }}
]

- KHÔNG giải thích.
- Nếu không thể trả JSON đúng, trả [].
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("LLM call error:", e)
        return results

    cleaned = content.replace("```json", "").replace("```", "").strip()

    try:
        ranking = json.loads(cleaned) if cleaned else []
        if not isinstance(ranking, list):
            return results
    except Exception:
        return results

    # Nếu ranking rỗng: giữ nguyên order retrieval
    if not ranking:
        return results

    # default fields
    for h in candidates:
        h["rerank_score"] = 0.0
        h["include_in_context"] = False

    # attach fields theo ranking
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if not (0 <= di < len(candidates)):
                continue
            candidates[di]["rerank_score"] = float(item.get("score", 0.0))
            candidates[di]["include_in_context"] = _parse_bool(item.get("include_in_context", False))
        except Exception:
            continue

    # reorder theo ranking (stable)
    used = set()
    reranked = []
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if 0 <= di < len(candidates) and di not in used:
                reranked.append(candidates[di])
                used.add(di)
        except Exception:
            continue

    # add missing
    for i, h in enumerate(candidates):
        if i not in used:
            reranked.append(h)

    # append tail
    if len(results) > len(candidates):
        reranked.extend(results[len(candidates):])

    docs_for_log = []
    return reranked, docs_for_log
