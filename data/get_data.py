import json
import requests

URL = "https://datasets-server.huggingface.co/rows"
TARGET = 150

def to_difficulty(level: str) -> str:
    level = (level or "").lower()
    if level in {"easy", "medium", "hard"}:
        return level
    return "medium"

def map_hotpot_row(raw: dict, idx: int) -> dict:
    ctx = raw.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])

    mapped_context = []
    for i, title in enumerate(titles):
        sent_list = sentences[i] if i < len(sentences) else []
        if isinstance(sent_list, list):
            text = " ".join(s.strip() for s in sent_list if isinstance(s, str)).strip()
        else:
            text = str(sent_list).strip()

        mapped_context.append({
            "title": str(title).strip(),
            "text": text
        })

    return {
        "qid": f"hp{idx + 1}",
        "difficulty": to_difficulty(raw.get("level", "")),
        "question": str(raw.get("question", "")).strip(),
        "gold_answer": str(raw.get("answer", "")).strip(),
        "context": mapped_context
    }

all_raw = []
offset = 0

while len(all_raw) < TARGET:
    batch_size = min(100, TARGET - len(all_raw))  # /rows cho toi da 100
    params = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "offset": offset,
        "length": batch_size
    }

    res = requests.get(URL, params=params, timeout=60)
    res.raise_for_status()

    rows = res.json().get("rows", [])
    if not rows:
        break

    all_raw.extend(item.get("row", {}) for item in rows)
    offset += len(rows)

mapped = [map_hotpot_row(r, i) for i, r in enumerate(all_raw)]

with open("hotpot_150.json", "w", encoding="utf-8") as f:
    json.dump(mapped, f, ensure_ascii=False, indent=2)

print(f"Da luu {len(mapped)} mau vao hotpot_150.json")