import time
import re
import unicodedata
from pathlib import Path
from datetime import datetime


def _sanitize_filename(text: str, max_len: int = 100) -> str:
    """
    Biến query thành tên file an toàn:
    - bỏ ký tự cấm trên Windows
    - normalize unicode
    - thay khoảng trắng bằng _
    - cắt ngắn để tránh path quá dài
    """
    if not text:
        return "empty_query"

    # normalize unicode về dạng an toàn
    text = unicodedata.normalize("NFKD", text)

    # chuyển về lowercase
    text = text.lower().strip()

    # bỏ ký tự cấm trong filename của Windows
    text = re.sub(r'[\\/:*?"<>|]', '', text)

    # thay mọi khoảng trắng bằng _
    text = re.sub(r'\s+', '_', text)

    # bỏ các ký tự không alnum còn sót lại (tùy chọn an toàn thêm)
    text = re.sub(r'[^a-zA-Z0-9_\-]', '', text)

    # giới hạn độ dài
    return text[:max_len]


class TimingLog:

    def __init__(self, query: str):
        self.query = query
        self.start = time.time()
        self.marks = []
        self.sub_steps = {}

    def mark(self, name: str):
        self.marks.append((name, time.time()))

    def mark_sub(self, group: str, name: str):
        self.sub_steps.setdefault(group, []).append((name, time.time()))

    def finish(self, log_enabled: bool = True):
        if not log_enabled:
            return

        base = Path("debug_timing")
        base.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 🔧 SANITIZE QUERY TRƯỚC KHI DÙNG LÀM TÊN FILE
        safe = _sanitize_filename(self.query)

        path = base / f"{ts}__{safe}.txt"

        lines = []
        lines.append(f"QUERY: {self.query}\n")

        prev = self.start

        for name, t in self.marks:
            dt = t - prev
            lines.append(f"- {name}: {dt:.3f}s")
            prev = t

        for group, steps in self.sub_steps.items():
            lines.append(f"\n[{group}]")
            g_prev = None
            for name, t in steps:
                if g_prev is None:
                    g_prev = t
                dt = t - g_prev
                lines.append(f"  - {name}: {dt:.3f}s")
                g_prev = t

        total = time.time() - self.start
        lines.append(f"\nTOTAL: {total:.3f}s")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[TIMING] log written to: {path}")

        except Exception as e:
            # Nếu vì lý do gì vẫn lỗi path, fallback an toàn
            fallback = base / f"{ts}__fallback.txt"
            with open(fallback, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[TIMING] failed to write original path, used fallback: {fallback}")
            print(f"[TIMING] reason: {e}")
