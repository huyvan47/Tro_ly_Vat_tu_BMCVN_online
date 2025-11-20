import numpy as np
from openai import OpenAI

client = OpenAI(api_key="...")

data = np.load("tong-hop-data-phong-vat-tu.npz", allow_pickle=True)
EMBS = data["embeddings"]          # (N, d)
QUESTIONS = data["questions"]      # (N,)
ANSWERS = data["answers"]          # (N,)

def embed_query(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def normalize_query(q: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": """
Bạn là Query Normalizer.
Nhiệm vụ:
- Sửa lỗi chính tả
- Chuẩn hoá câu
- Giữ nguyên ý nghĩa
- Trả về chuỗi văn bản đã chuẩn hoá
"""},
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()

def search(query: str, top_k: int = 5):
    norm_query = normalize_query(query)
    print("norm_query: ", norm_query)
    vq = embed_query(norm_query)  # (d,)
    sims = EMBS @ vq         # (N,) – cosine similarity
    idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idx:
        results.append({
            "question": str(QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        })
    return results

if __name__ == "__main__":
    q = "kế hoach vât tư hằng ngày"
    hits = search(q, top_k=5)
    print("Câu truy vấn: ", q)
    for h in hits:
        print(f"{h['score']:.4f} | {h['question']} -> {h['answer'][:50]}...")
