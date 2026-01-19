import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List


def _safe_folder_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\d]+", "_", s)
    return s[:80]


def write_multi_query_logs(
    *,
    original_query: str,
    subs: list,
    results_by_query: list,
    fused_hits: Optional[List] = None
):
    """
    Ghi log multi-query ra thư mục riêng cho từng query gốc.
    """

    base = Path("debug_multi")
    base.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    qname = _safe_folder_name(original_query)

    root = base / f"{ts}__{qname}"
    root.mkdir(parents=True, exist_ok=True)

    # 1) query gốc
    with open(root / "query.txt", "w", encoding="utf-8") as f:
        f.write(original_query)

    # 2) sub-queries
    with open(root / "sub_queries.json", "w", encoding="utf-8") as f:
        json.dump(subs, f, ensure_ascii=False, indent=2)

    # 3) thống kê retrieval
    stats_lines = []
    for i, r in enumerate(results_by_query):
        stats_lines.append(
            f"[{i}] purpose={r['purpose']} | hits={len(r.get('hits') or [])}"
        )

    with open(root / "retrieval_stats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))

    # 4) chi tiết từng sub-query
    subs_dir = root / "subs"
    subs_dir.mkdir(exist_ok=True)

    for i, r in enumerate(results_by_query):
        purpose = r.get("purpose", "general")
        hits = r.get("hits") or []

        # Lấy text của sub-query tương ứng
        sub_q = subs[i].get("q") if i < len(subs) else ""

        out = {
            "purpose": purpose,
            "query": sub_q,  
            "num_hits": len(hits),
            "docs": hits,
        }

        fname = f"sub_{i}_{purpose}.json"
        with open(subs_dir / fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # 5) fused summary
    if fused_hits:
        summary = {
            "num_fused_hits": len(fused_hits),
            "top_20": fused_hits[:20],
        }
        with open(root / "fused_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[MULTI-QUERY DEBUG] Logs written to: {root}")
