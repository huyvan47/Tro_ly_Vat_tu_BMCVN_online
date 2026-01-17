from openai import OpenAI
from rag.config import RAGConfig
from rag.kb_loader import load_npz
from rag.logger_csv import append_log_to_csv
from rag.pipeline import answer_with_suggestions
from policies.v7_policy import PolicyV7 as policy
from pathlib import Path
from rag.debug_log import debug_log
import traceback

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_TXT = BASE_DIR / "questions.txt"
CSV_PATH = "rag_logs.csv"
KB = "data-kd-1-4-16-1-2026.npz"
OPENAI_KEY = "..."

def iter_questions(txt_path: str):
    """
    Đọc file txt, yield từng câu hỏi:
    - bỏ dòng trống
    - bỏ comment nếu dòng bắt đầu bằng '#'
    - strip khoảng trắng
    """
    print('txt_path:', txt_path)
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p.resolve()}")

    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.strip()
        if not q:
            continue
        if q.startswith("#"):
            continue
        yield q


def run_batch_questions(KB, API_KEY):
    # 1) đọc query từ CLI

    # 2) init OpenAI client (đặt key theo env là tốt nhất)
    client = OpenAI(api_key=API_KEY)

    # 3) load KB (1 lần)
    # kb = load_npz("data-kd-nam-benh-full-fix-noise.npz")
    kb = load_npz(KB)

    cfg = RAGConfig()

    for i, q in enumerate(iter_questions(QUESTIONS_TXT), start=1):
        debug_log(f"[{i}] Q: {q}")

        res = answer_with_suggestions(
            user_query=q,
            kb=kb,
            client=client,
            cfg=cfg,
            policy=policy,
        )

        append_log_to_csv(
            csv_path=CSV_PATH,
            user_query=q,
            norm_query=res.get("norm_query", ""),
            context_build=res.get("context", ""),
            strategy=res.get("strategy", ""),
            prof=res.get("profile", {}) or {},
            res=res,
            route=res.get("route", "RAG"),
        )

    print(f"\nHoàn tất. Log đã ghi vào: {CSV_PATH}")

def main(KB, API_KEY):
    # 1) đọc query từ CLI

    # 2) init OpenAI client (đặt key theo env là tốt nhất)
    client = OpenAI(api_key=API_KEY)

    # 3) load KB (1 lần)
    # kb = load_npz("data-kd-nam-benh-full-fix-noise.npz")
    kb = load_npz(KB)

    cfg = RAGConfig()

    while True:
        try:
            q = input("Query: ").strip()
            if not q:
                break
        except EOFError:
            print("EOF (stdin closed).")
            break
        except KeyboardInterrupt:
            print("KeyboardInterrupt.")
            break

        try:            
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
                context_build=res.get("context_build", ""),
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

        except Exception as e:
            print("Unhandled exception in loop: ", e)
            traceback.print_exc()
            continue

if __name__ == "__main__":
    # # Test nhiều câu hỏi
    # run_batch_questions()

    # Test một câu hỏi
    main(KB, OPENAI_KEY)

