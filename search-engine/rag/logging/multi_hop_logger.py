import json
from pathlib import Path
from datetime import datetime
import re


def _safe_folder_name(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\d]+", "_", text)
    return text[:max_len]


def write_multi_hop_logs(
    *,
    original_query: str,
    hops_data: list,
    final_hits: list,
):
    """
    Ghi log chi tiết cho multi-hop

    hops_data = [
        {
            "hop": 1,
            "query": "...",
            "num_hits": 42,
            "hits": [...],
            "decision": {
                "need_next_hop": true,
                "next_query": "...",
                "reason": "..."
            }
        },
        ...
    ]
    """

    base = Path("debug_multi_hop")
    base.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    qname = _safe_folder_name(original_query)

    root = base / f"{ts}__{qname}"
    root.mkdir(parents=True, exist_ok=True)

    # 1) query gốc
    with open(root / "query.txt", "w", encoding="utf-8") as f:
        f.write(original_query)

    # 2) tổng quan các hop
    summary = []

    for h in hops_data:
        summary.append({
            "hop": h.get("hop"),
            "query": h.get("query"),
            "num_hits": h.get("num_hits"),
            "decision": h.get("decision"),
        })

    with open(root / "hops_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 3) ghi từng hop ra file riêng
    hops_dir = root / "hops"
    hops_dir.mkdir(exist_ok=True)

    for h in hops_data:
        hop = h.get("hop")
        fname = f"hop_{hop}.json"

        out = {
            "hop": hop,
            "query": h.get("query"),
            "num_hits": h.get("num_hits"),
            "decision": h.get("decision"),
            "docs": h.get("hits"),
        }

        with open(hops_dir / fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # 4) final fused hits
    final = {
        "num_final_hits": len(final_hits),
        "top_30": final_hits[:30],
    }

    with open(root / "final_hits.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"[MULTI-HOP DEBUG] Logs written to: {root}")
