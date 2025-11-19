from openai import OpenAI
client = OpenAI(api_key="sk-proj-........")

FT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::CdBoxNIT"

resp = client.chat.completions.create(
    model=FT_MODEL,
    temperature=0,
    messages=[
        {"role":"system","content":"Bạn là Trợ lý Vật tư BMCVN. Giọng chuyên nghiệp, trả lời ngắn gọn, có bullet khi cần."},
        {"role":"user","content":"CHA240-TEEN"}
    ]
)
print(resp.choices[0].message.content)