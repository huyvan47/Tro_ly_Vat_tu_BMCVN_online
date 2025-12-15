def normalize_query(client, q: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
Bạn là Query Normalizer.
Không thay đổi các mã sản phẩm hoặc hoạt chất như Kenbast 15SL, glufosinate_amonium, ...
Chỉ sửa lỗi chính tả và chuẩn hoá văn bản.
""".strip()
            },
            {"role": "user", "content": q}
        ],
    )
    return resp.choices[0].message.content.strip()