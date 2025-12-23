import os
import pandas as pd
import json
from datetime import datetime

def append_log_to_csv(
    csv_path: str,
    user_query: str,
    norm_query: str,
    strategy: str,
    prof: dict,
    res: dict,
    route: str = "RAG",
):
    """
    Append 1 dòng log vào CSV (an toàn khi file đang được mở để xem).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": ts,
        "route": route,
        "user_query": user_query,
        "norm_query": norm_query,
        "strategy": strategy,
        "top1": float(prof.get("top1", 0.0)),
        "top2": float(prof.get("top2", 0.0)),
        "gap": float(prof.get("gap", 0.0)),
        "mean5": float(prof.get("mean5", 0.0)),
        "n": int(prof.get("n", 0)),
        "conf": float(prof.get("conf", 0.0)),
        "answer_text": res.get("text", ""),
        "img_keys": json.dumps(res.get("img_keys", []), ensure_ascii=False),

        # "docs_json": json.dumps(res.get("docs", []), ensure_ascii=False),
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)

    # append mode
    df.to_csv(
        csv_path,
        mode="a",
        index=False,
        header=header,
        encoding="utf-8-sig",
    )
    return csv_path