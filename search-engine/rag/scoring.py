import numpy as np

# ====== CẤU HÌNH TRỌNG SỐ ======
# Có thể tune lại theo dataset thực tế
TAG_WEIGHT = 0.65
EMBED_WEIGHT = 0.35

# Số tag tối đa có thể khớp trong hệ của bạn (tùy schema)
# Ví dụ: crop + pest + entity + disease_group ...
MAX_TAG_POSSIBLE = 5.0


def fused_score(h: dict) -> float:
    """
    Score hợp nhất ưu tiên tuyệt đối tag_score.
    Không dùng rerank_score nữa.

    Công thức:
      fused = TAG_WEIGHT * normalized_tag + EMBED_WEIGHT * embedding_score

    - tag_score càng cao -> càng quan trọng
    - embedding chỉ đóng vai trò phụ để phân biệt khi tag ngang nhau
    """

    # Lấy các thành phần
    t = float(h.get("tag_score", 0.0) or 0.0)
    e = float(h.get("score", 0.0) or 0.0)

    # Normalize tag_score về 0-1
    t_norm = min(1.0, t / MAX_TAG_POSSIBLE)

    return TAG_WEIGHT * t_norm + EMBED_WEIGHT * e


def analyze_hits_fused(hits: list) -> dict:
    """
    Profile chất lượng tập kết quả dựa trên:
      - fused_score (đã ưu tiên tag)
      - độ chênh lệch top1/top2
      - độ dày evidence (mean5)
      - độ phủ tag (tag_density)

    Dùng để quyết định chiến lược trả lời.
    """

    if not hits:
        return {
            "top1": 0.0,
            "top2": 0.0,
            "gap": 0.0,
            "mean5": 0.0,
            "n": 0,
            "conf": 0.0,
            "tag_density": 0.0,
        }

    # Tính fused_score cho từng hit nếu chưa có
    for h in hits:
        if "fused_score" not in h:
            h["fused_score"] = fused_score(h)

    # Lấy danh sách score đã sort
    raw_scores = [float(h.get("fused_score", 0.0) or 0.0) for h in hits]
    scores = sorted(raw_scores, reverse=True)

    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2

    mean5 = (
        float(np.mean(scores[:5]))
        if len(scores) >= 5
        else float(np.mean(scores))
    )

    n = len(scores)

    # ===== TÍNH CONFIDENCE =====

    # Bonus theo gap – nhưng không hard-zero
    bonus = min(1.0, max(0.05, gap / 0.15))

    # Density của fused_score
    density = mean5 / max(top1, 1e-6)
    density = min(1.0, max(0.0, density))

    base_conf = top1 * bonus * (0.5 + 0.5 * density)

    # ===== THÊM THÀNH PHẦN TAG =====

    # Độ dày tag của top 5 doc
    tag_scores = [float(h.get("tag_score", 0.0) or 0.0) for h in hits]
    tag_scores_sorted = sorted(tag_scores, reverse=True)

    tag_mean5 = (
        float(np.mean(tag_scores_sorted[:5]))
        if len(tag_scores_sorted) >= 5
        else float(np.mean(tag_scores_sorted))
    )

    tag_density = min(1.0, tag_mean5 / MAX_TAG_POSSIBLE)

    # Tăng conf nếu top docs đều khớp tag tốt
    conf = base_conf * (0.6 + 0.4 * tag_density)

    return {
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "mean5": mean5,
        "n": n,
        "conf": conf,
        "tag_density": tag_density,
    }


def analyze_hits(hits: list) -> dict:
    """
    Bản rút gọn dùng cho single-query (nếu cần).
    Không dùng rerank.
    """

    if not hits:
        return {
            "top1": 0.0,
            "top2": 0.0,
            "gap": 0.0,
            "mean5": 0.0,
            "n": 0,
            "tag_density": 0.0,
        }

    # Tính fused_score trước
    for h in hits:
        if "fused_score" not in h:
            h["fused_score"] = fused_score(h)

    scores = [float(h.get("fused_score", 0.0)) for h in hits]

    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2

    mean5 = (
        float(np.mean(scores[:5]))
        if len(scores) >= 5
        else float(np.mean(scores))
    )

    # Tag density đơn giản
    tag_scores = [float(h.get("tag_score", 0.0) or 0.0) for h in hits]
    tag_mean = float(np.mean(tag_scores)) if tag_scores else 0.0
    tag_density = min(1.0, tag_mean / MAX_TAG_POSSIBLE)

    return {
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "mean5": mean5,
        "n": len(hits),
        "tag_density": tag_density,
    }
