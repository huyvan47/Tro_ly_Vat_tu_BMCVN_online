# -*- coding: utf-8 -*-
import os, sys, json, itertools
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, ASSISTANT_ID, MAX_FILE_IDS_PER_MESSAGE

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "uploaded_files.json")

def chunks(lst, n):
    it = iter(lst)
    while True:
        batch = list(next(it, None) for _ in range(n))
        batch = [x for x in batch if x is not None]
        if not batch:
            break
        yield batch

def ensure_assistant(client: OpenAI) -> str:
    if ASSISTANT_ID:
        return ASSISTANT_ID
    a = client.beta.assistants.create(model=CHAT_MODEL, tools=[{"type":"file_search"}])
    return a.id

def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/answer_online.py "Câu hỏi..."')
        sys.exit(0)

    question = sys.argv[1]

    client = OpenAI(api_key=OPENAI_API_KEY)
    assistant_id = ensure_assistant(client)

    # Tải danh sách file_id đã upload
    file_ids = []
    if os.path.exists(STATE_FILE):
        state = json.load(open(STATE_FILE, "r", encoding="utf-8"))
        for rec in state.values():
            if isinstance(rec, dict) and rec.get("file_id"):
                file_ids.append(rec["file_id"])

    # Tạo thread
    thread = client.beta.threads.create()

    # Gắn file_ids theo lô để tránh payload quá lớn
    def batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    attachments_batches = []
    if file_ids:
        for batch in batches(file_ids, MAX_FILE_IDS_PER_MESSAGE):
            attachments_batches.append([{"file_id": fid, "tools":[{"type":"file_search"}]} for fid in batch])

    # Message đầu: câu hỏi + gắn lô đầu tiên (nếu có)
    first_attachments = attachments_batches[0] if attachments_batches else None
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question,
        attachments=first_attachments
    )

    # Các lô còn lại (nếu có) sẽ được thêm vào như message rỗng để đăng ký file
    for extra in attachments_batches[1:]:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="(attach files)",
            attachments=extra
        )

    # Run: assistant sẽ tự retrieval từ các file đã attach
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    # Poll đến khi xong
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if r.status in ("completed","failed","cancelled","expired"):
            break

    if r.status != "completed":
        print(f"Run status: {r.status}")
        sys.exit(1)

    # Lấy câu trả lời mới nhất
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    for m in reversed(msgs.data):
        if m.role == "assistant":
            out = []
            for part in m.content:
                if part.type == "text":
                    out.append(part.text.value)
            print("\n".join(out))
            break

if __name__ == "__main__":
    main()
