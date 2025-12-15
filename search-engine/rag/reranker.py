from rag.config import RAGConfig
import json

def llm_rerank(client, norm_query: str, results: list, top_k_rerank: int = RAGConfig.top_k_rerank):
    """
    Rerank top_k_rerank documents bằng LLM mini.
    Sử dụng include_in_context để ưu tiên doc đưa vào context.
    LUÔN trả về list hợp lệ (fallback nếu lỗi).
    """
    if not results or len(results) == 1:
        return results

    candidates = results[:top_k_rerank]

    # Build docs block
    doc_texts = []
    for i, h in enumerate(candidates):
        ans = str(h.get("answer", ""))
        if len(ans) > RAGConfig.rerank_snippet_chars:
            ans = ans[:RAGConfig.rerank_snippet_chars] + " ..."
        block = (
            f"[DOC {i}]\n"
            f"QUESTION: {h.get('question','')}\n"
            f"ALT_QUESTION: {h.get('alt_question','')}\n"
            f"ANSWER_SNIPPET:\n{ans}"
        )
        doc_texts.append(block)

    docs_block = "\n\n------------------------\n\n".join(doc_texts)
    if RAGConfig.debug_rerank:
        print("docs_block:\n", docs_block)

    system_prompt = (
        "Bạn là LLM dùng để RERANK tài liệu cho hệ thống hỏi đáp bệnh cây / nông nghiệp BMCVN. "
        "Bạn CHỈ được trả về JSON THUẦN, không có code block hay markdown."
    )

    user_prompt = f"""
CÂU HỎI:
\"\"\"{norm_query}\"\"\"


CÁC TÀI LIỆU ỨNG VIÊN:

{docs_block}

YÊU CẦU:
- Với mỗi DOC, hãy:
  (1) Chấm điểm liên quan từ 0–1
  (2) Quyết định có nên đưa DOC này vào NGỮ CẢNH trả lời hay không bằng true/false
- Ưu tiên đưa vào NGỮ CẢNH các DOC thuộc các nhóm:
  - Cách trị / thuốc
  - Triệu chứng nhận biết
  - Nguyên nhân & lây lan
  - Biện pháp canh tác / cảnh báo

- Trả về DUY NHẤT JSON, ví dụ:
[
  {{ "doc_index": 0, "score": 0.92, "include_in_context": true }},
  {{ "doc_index": 1, "score": 0.30, "include_in_context": false }}
]
- KHÔNG dùng ```json hay markdown.
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
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM call error:", e)
        return results

    cleaned = content.replace("```json", "").replace("```", "").strip()

    try:
        ranking = json.loads(cleaned)
        if not isinstance(ranking, list):
            raise ValueError("JSON is not a list")
    except Exception:
        # fallback nếu parse fail
        return results

    idx_to_result = {i: candidates[i] for i in range(len(candidates))}

    # default
    for h in candidates:
        h["rerank_score"] = 0.0
        h["include_in_context"] = False

    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if di not in idx_to_result:
                continue
            score = float(item.get("score", 0))
            include_flag = bool(item.get("include_in_context", False))

            idx_to_result[di]["rerank_score"] = score
            idx_to_result[di]["include_in_context"] = include_flag
        except Exception:
            continue

    # reorder theo thứ tự LLM
    used = set()
    reranked = []
    for item in ranking:
        try:
            di = int(item.get("doc_index", -1))
            if di in idx_to_result and di not in used:
                reranked.append(idx_to_result[di])
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

    # ưu tiên include_in_context lên đầu
    selected = [h for h in reranked if h.get("include_in_context") is True]
    others = [h for h in reranked if not h.get("include_in_context")]

    return (selected + others) if selected else reranked