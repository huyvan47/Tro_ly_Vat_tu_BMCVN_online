import numpy as np

def analyze_hits_fused(hits: list) -> dict:
    """
    Profile dựa trên fused_score thay vì embedding score.
    Trả về top1/top2/gap/mean5/conf.
    conf = top1 * min(1, gap/0.15)  (thưởng gap, phạt trường hợp top1 ~ top2)
    """
    if not hits:
        return {"top1": 0.0, "top2": 0.0, "gap": 0.0, "mean5": 0.0, "n": 0, "conf": 0.0}

    scores = [float(h.get("fused_score", 0.0)) for h in hits]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2
    mean5 = float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores))

    # 0.15 là "độ rộng" gap để đạt full bonus (bạn có thể tinh chỉnh theo dataset)
    conf = top1 * min(1.0, (gap / 0.15) if gap > 0 else 0.0)

    return {"top1": top1, "top2": top2, "gap": gap, "mean5": mean5, "n": len(hits), "conf": conf}

def analyze_hits(hits: list) -> dict:
    """
    Trả profile để quyết định chiến lược trả lời.
    """
    if not hits:
        return {"top1": 0.0, "top2": 0.0, "gap": 0.0, "mean5": 0.0, "n": 0}

    scores = [float(h.get("score", 0.0)) for h in hits]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2
    mean5 = float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores))
    return {"top1": top1, "top2": top2, "gap": gap, "mean5": mean5, "n": len(hits)}

def fused_score(h: dict, w_r: float = 0.70, w_e: float = 0.30) -> float:
    """
    Score hợp nhất cho 1 doc.
    - Ưu tiên rerank_score (nếu có)
    - Fallback sang embedding score nếu rerank_score không có/0
    """
    r = float(h.get("rerank_score", 0.0) or 0.0)
    e = float(h.get("score", 0.0) or 0.0)
    if r <= 0.0:
        return e
    return w_r * r + w_e * e