import numpy as np

import numpy as np

def analyze_hits_fused(hits: list) -> dict:
    """
    Profile dựa trên fused_score.

    Fix quan trọng:
    - KHÔNG lấy top1/top2 theo vị trí hits[0], hits[1] vì hits của bạn đang sort theo
      (tag_hits, mq_rrf, fused_score) -> có thể làm gap âm.
    - Thay vào đó: sort scores theo fused_score giảm dần để lấy top1/top2 đúng nghĩa.

    conf:
      - bonus theo gap (không hard-zero khi gap<=0)
      - thưởng thêm nếu evidence dày (mean5 gần top1)
    """
    if not hits:
        return {"top1": 0.0, "top2": 0.0, "gap": 0.0, "mean5": 0.0, "n": 0, "conf": 0.0}

    raw_scores = [float(h.get("fused_score", 0.0) or 0.0) for h in hits]
    scores = sorted(raw_scores, reverse=True)  # FIX: lấy top theo fused_score thật sự

    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2
    mean5 = float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores))
    n = len(scores)

    # 0.15 là "độ rộng" gap để đạt full bonus (có thể tune theo dataset)
    # FIX: không để conf=0 chỉ vì gap<=0 (do noise/sort mismatch)
    bonus = min(1.0, max(0.05, gap / 0.15))

    # thưởng "độ dày evidence": nếu mean5 gần top1 thì đáng tin hơn
    density = mean5 / max(top1, 1e-6)
    density = min(1.0, max(0.0, density))

    conf = top1 * bonus * (0.5 + 0.5 * density)

    return {"top1": top1, "top2": top2, "gap": gap, "mean5": mean5, "n": n, "conf": conf}



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