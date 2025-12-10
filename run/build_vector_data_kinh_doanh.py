import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent

# 👉 Sửa đúng file CSV bạn đang dùng
DATA = ROOT / "data/data-kinh-doanh/data-kinh-doanh_FIXED-3.csv"

# 👉 Tên file vector xuất ra
OUT_FILE = "data-kinh-doanh-nam-benh-full.npz"

# 👉 API KEY
# Nên để trong biến môi trường OPENAI_API_KEY thay vì ghi cứng
client = OpenAI(api_key="...")

# ==============================
#        LOAD CSV
# ==============================

df = pd.read_csv(DATA, encoding="utf-8")

# Ép kiểu an toàn
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)

# Nếu chưa có alt_questions thì tạo cột rỗng
if "alt_questions" not in df.columns:
    df["alt_questions"] = ""

# Nếu chưa có tags thì tạo cột rỗng
if "tags" not in df.columns:
    df["tags"] = ""

df["alt_questions"] = df["alt_questions"].astype(str)
df["tags"] = df["tags"].astype(str)

# ==============================
#   BUILD INPUT TEXTS FOR EMBED
# ==============================

inputs = []

for _, row in df.iterrows():
    q = row["question"].strip()
    a = row["answer"].strip()
    alt = row["alt_questions"].strip()
    tags = row["tags"].strip()

    parts = []

    if q:
        parts.append(f"Hỏi: {q}")

    if alt and alt.lower() != "nan":
        # Chuẩn hóa alt_questions dạng | thành câu hỏi tự nhiên
        alt_clean = alt.replace("|", ", ")
        parts.append(f"Cách hỏi khác: {alt_clean}")

    if tags and tags.lower() != "nan":
        parts.append(f"Từ khóa: {tags}")

    if a:
        parts.append(f"Trả lời: {a}")

    text = ". ".join(parts) + "."
    inputs.append(text)

print(f"🔢 Tổng số dòng cần embed: {len(inputs)}")

# ==============================
#       EMBEDDING (BATCH)
# ==============================

print("🚀 Bắt đầu embedding theo batch ...")

BATCH_SIZE = 200  # có thể chỉnh 100–300 tùy ý

all_embs = []

for start in range(0, len(inputs), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(inputs))
    batch = inputs[start:end]

    print(f"➡ Embedding batch {start} → {end - 1} (số lượng: {len(batch)})")

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch,
    )

    batch_embs = [item.embedding for item in resp.data]
    all_embs.extend(batch_embs)

# Chuyển sang numpy
embs = np.array(all_embs, dtype=np.float32)

# Kiểm tra an toàn: số vector == số dòng
assert embs.shape[0] == len(df), f"Mismatch: {embs.shape[0]} embeddings nhưng {len(df)} dòng CSV"

# ✅ Chuẩn hoá vector đơn vị (cosine similarity chuẩn)
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
embs = embs / norms

print("🔥 Embedding xong. Tổng số vector:", len(embs))

# ==============================
#        SAVE NPZ
# ==============================

np.savez(
    OUT_FILE,
    embeddings=embs,

    # ⚠️ id của bạn là string → để dtype=object
    ids=df["id"].to_numpy(dtype=object),

    questions=df["question"].to_numpy(dtype=object),
    answers=df["answer"].to_numpy(dtype=object),
    alt_questions=df["alt_questions"].to_numpy(dtype=object),
    category=df["category"].to_numpy(dtype=object),
    tags=df["tags"].to_numpy(dtype=object),
)

print(f"✅ ĐÃ BUILD XONG VECTOR FILE → {OUT_FILE}")
print(f"✅ Tổng số vector: {len(embs)}")
