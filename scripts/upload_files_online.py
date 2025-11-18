# upload_files_online.py
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from pathlib import Path
import os, json, time, hashlib, sys

client = OpenAI(api_key="sk-........")

# ===== 1) Đường dẫn chuẩn, KHÔNG ghi đè IMG_DIR 2 lần =====
BASE_DIR = Path(__file__).resolve().parents[1]   # ...\Tro_ly_Vat_tu_BMCVN_online
IMG_DIR  = BASE_DIR / "images"                   # ảnh để upload
OUT_JSON = BASE_DIR / "uploaded_images.json"     # mapping path -> {file_id, fingerprint,...}

ALLOWED     = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv"}
PACING      = 0.25     # nghỉ giữa lượt để tránh rate limit
MAX_RETRIES = 5

def fp(p: Path) -> str:
    """Fingerprint để resume: path + mtime + size."""
    st = p.stat()
    s = f"{p.resolve()}::{int(st.st_mtime)}::{st.st_size}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_state():
    """Đọc uploaded_images.json. Hỗ trợ cả dạng list (cũ) và dict (mới)."""
    if not OUT_JSON.exists():
        return {}
    try:
        raw = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        # Nếu là list cũ: [{path, file_id, fingerprint?}, ...] -> chuyển sang dict {abs_path: {...}}
        if isinstance(raw, list):
            new_state = {}
            for item in raw:
                if not isinstance(item, dict): 
                    continue
                path = item.get("path")
                fid  = item.get("file_id")
                if not path or not fid: 
                    continue
                key = str(Path(path).resolve())
                # Nếu chưa có fingerprint trong list cũ, sẽ tính lại khi file còn tồn tại
                new_state[key] = {
                    "path": path,
                    "file_id": fid,
                    "fingerprint": item.get("fingerprint", None),
                    "uploaded_at": item.get("uploaded_at", None),
                }
            return new_state
        elif isinstance(raw, dict):
            return raw
        else:
            return {}
    except Exception:
        return {}

def save_state(state: dict):
    OUT_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def upload_one(p: Path) -> str:
    """Upload 1 file với retry + exponential backoff."""
    delay = PACING
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(p, "rb") as f:
                obj = client.files.create(file=f, purpose="assistants")
            return obj.id
        except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
            # In log lỗi để debug
            print(f"[{attempt}/{MAX_RETRIES}] Upload lỗi {p.name}: {e}", file=sys.stderr)
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0) + 0.05 * attempt

def main():
    print("Project root:", BASE_DIR)
    print("Images dir  :", IMG_DIR, "| exists:", IMG_DIR.exists())

    if not IMG_DIR.exists():
        print(f"ERROR: Không tìm thấy thư mục images ở: {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in sorted(IMG_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED]
    if not files:
        print(f"Không có file hợp lệ trong: {IMG_DIR} (đuôi: {sorted(ALLOWED)})")
        sys.exit(0)

    state = load_state()
    uploaded, skipped = 0, 0

    for p in files:
        key = str(p.resolve())
        fingerprint = fp(p)
        rec = state.get(key)

        # Nếu đã có và fingerprint khớp => bỏ qua
        if rec and rec.get("fingerprint") == fingerprint and rec.get("file_id"):
            skipped += 1
            continue

        try:
            fid = upload_one(p)
            state[key] = {
                "path": str(p),
                "file_id": fid,
                "fingerprint": fingerprint,
                "uploaded_at": int(time.time()),
            }
            uploaded += 1
            print(f"✅ Uploaded {p.name} → {fid}")
            time.sleep(PACING)
        except Exception as e:
            state[key] = {
                "path": str(p),
                "error": str(e),
                "fingerprint": fingerprint,
                "uploaded_at": int(time.time()),
            }
            print(f"❌ FAILED {p.name}: {e}", file=sys.stderr)

        save_state(state)

    save_state(state)
    print(f"\nDone! Uploaded: {uploaded}, Skipped: {skipped}, Tracked: {len(state)} → {OUT_JSON}")

if __name__ == "__main__":
    main()
