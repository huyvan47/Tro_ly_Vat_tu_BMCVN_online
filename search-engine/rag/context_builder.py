def build_context_from_hits(hits_for_ctx: list) -> str:
    blocks = []
    for i, h in enumerate(hits_for_ctx, 1):
        block = (
            f"[DOC {i}]\n"
            f"CÂU HỎI: {h.get('question','')}\n"
            f"HỎI KHÁC: {h.get('alt_question','')}\n"
            f"NỘI DUNG:\n{h.get('answer','')}"
        )
        # print('block: ', block)
        blocks.append(block)
    return "\n\n--------------------\n\n".join(blocks)

def choose_adaptive_max_ctx(hits_reranked, is_listing: bool = False):
    # dùng fused_score (ổn định cả khi rerank bật/tắt)
    scores = [float(h.get("fused_score", h.get("rerank_score", 0.0) or 0.0)) for h in hits_reranked[:4]]
    scores += [0] * (4 - len(scores))
    s1, s2, s3, s4 = scores

    if is_listing:
        if s1 >= 0.75 and s2 >= 0.65 and s3 >= 0.55:
            return 60
        if s1 >= 0.65 and s2 >= 0.55:
            return 50
        return 40

    if s1 >= 0.90 and s2 >= 0.80 and s3 >= 0.75 and s4 >= 0.70:
        return 32
    if s1 >= 0.85 and s2 >= 0.75 and s3 >= 0.70:
        return 28
    if s1 >= 0.80 and s2 >= 0.65:
        return 24

    return 24