from openai import OpenAI

from rag.config import RAGConfig
from rag.kb_loader import load_npz
from rag.logger_csv import append_log_to_csv
from rag.pipeline import answer_with_suggestions
from policies.v7_policy import PolicyV7 as policy

def main():
    # 1) đọc query từ CLI

    # 2) init OpenAI client (đặt key theo env là tốt nhất)
    client = OpenAI(api_key="...")

    # 3) load KB (1 lần)
    kb = load_npz("data-kd-nam-benh-full-fix-noise.npz")

    cfg = RAGConfig()

    while True:
        q = input("Query: ").strip()
        if not q:
            break

        res = answer_with_suggestions(
            user_query=q,
            kb=kb,
            client=client,
            cfg=cfg,
            policy=policy,
        )

        # 5) log CSV
        csv_path = "rag_logs.csv"
        append_log_to_csv(
            csv_path=csv_path,
            user_query=q,
            norm_query=res.get("norm_query", ""),
            strategy=res.get("strategy", ""),
            prof=res.get("profile", {}) or {},
            res=res,
            route=res.get("route", "RAG"),
            # bạn có thể thêm policy_version nếu có
        )

        # 6) in kết quả
        print("\n===== KẾT QUẢ =====\n")
        print(res["text"])
        print("\nIMG_KEY:")
        print(res["img_keys"])
        print("\nSaved log to:", csv_path)


if __name__ == "__main__":
    main()
