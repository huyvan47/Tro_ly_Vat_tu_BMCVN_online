import json

with open("vattuphu-split.jsonl", "r", encoding="utf-8") as f, open("train_view.json", "w", encoding="utf-8") as out:
    data = [json.loads(line) for line in f]
    json.dump(data, out, indent=2, ensure_ascii=False)
