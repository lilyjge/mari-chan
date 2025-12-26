import json
import random
from datetime import datetime, timedelta
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def token_len(messages):
    return len(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    ))

def parse_time(ts) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

def load_messages(path, speaker):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
        msgs = []
        for m in data:
            msgs.append({
                "speaker": speaker,
                "timestamp": parse_time(m["Timestamp"]),
                "content": m["Contents"].strip(),
                "has_attachment": bool(m["Attachments"])
            })
    return msgs

# Load & merge
msgs: list = load_messages("data/l.json", "assistant") + load_messages("data/j.json", "user")
msgs.sort(key=lambda m: m["timestamp"])

# Filter attachments + neighbors
DROP_RADIUS = 2
drop_idxs = set()
for i, m in enumerate(msgs):
    if m["has_attachment"] or "https" in m["content"] or m["content"] == "":
        for j in range(max(0, i-DROP_RADIUS), min(len(msgs), i+DROP_RADIUS+1)):
            drop_idxs.add(j)

msgs = [m for i,m in enumerate(msgs) if i not in drop_idxs]

# Collapse into turns
turns: list = [] # {speaker, content, start time, end time}
for m in msgs: # for every message
    if (turns and m["speaker"] == turns[-1]["speaker"] and 
        m["timestamp"] - turns[-1]["end_time"] < timedelta(minutes=10)):
        turns[-1]["content"] += "\n" + m["content"]
        turns[-1]["end_time"] = m["timestamp"]
    else:
        newturn = m
        newturn["start_time"] = m["timestamp"]
        newturn["end_time"] = m["timestamp"]
        newturn.pop("timestamp")
        newturn.pop("has_attachment")
        turns.append(newturn)

# Split into conversations
convos: list = []
current = [turns[0]]
for t in turns[1:]:
    # if current last turn and new turn are more than hr apart, new convo
    if t["start_time"] - current[-1]["end_time"] > timedelta(hours=1) or t["speaker"] == current[-1]["speaker"]:
        convos.append(current)
        current = [t]
    else:
        current.append(t)
convos.append(current)

CONTEXT_SIZES = [1, 2, 3, 4, 5, 6, 8, 12]
PROBS = [0.3, 0.2, 0.2, 0.1, 0.07, 0.06, 0.04, 0.03]
examples: list = [] # [[{speaker, content}, ]]
length = []
for convo in convos:
    cur_len = sum(len(turn["content"].split()) for turn in convo)
    # cur_len = len(convo)
    length.append(cur_len)
    valid = [
        i for i in range(1, len(convo))
        if convo[i]["speaker"] == "assistant"
        and convo[i-1]["speaker"] == "user"
    ]
    if not valid:
        continue

    if len(convo) < 5:
        S = 1
    elif len(convo) < 12:
        S = 2
    else:
        S = 3

    indices = random.sample(valid, k=min(S, len(valid)))
    for idx in indices:
        k = random.choices(CONTEXT_SIZES, PROBS)[0] # length of context
        if idx - k < 0:
            k = idx
        while idx > k and convo[idx - k]["speaker"] != "user":
            k += 1
        if idx >= k and convo[idx - k]["speaker"] != "user":
            continue
        ctx: list = convo[idx-k:idx + 1]
        if ctx[0]["speaker"] != "user":
            print("weird idx and k ", idx, k)
        examples.append(ctx)

# minimum, q1, median, q3, maximum = np.percentile(length, [0, 25, 50, 75, 100])
# print("conversations length distributions words")
# print(f"Minimum: {minimum}")
# print(f"Q1 (25th percentile): {q1}")
# print(f"Median (50th percentile): {median}")
# print(f"Q3 (75th percentile): {q3}")
# print(f"Maximum: {maximum}")

length = []
with open("data/munged.jsonl", "w") as out:
    for ex in examples:
        messages = [{"role": turn["speaker"], "content": turn["content"]} for turn in ex]
        if token_len(messages) > 768:
            continue
        cur_len = sum(len(turn["content"].split()) for turn in ex)
        length.append(cur_len)
        line = json.dumps({"messages": messages})
        out.write(line + "\n")

minimum, q1, median, q3, maximum = np.percentile(length, [0, 25, 50, 75, 100])
print("samples length distributions words")
print(f"Minimum: {minimum}")
print(f"Q1 (25th percentile): {q1}")
print(f"Median (50th percentile): {median}")
print(f"Q3 (75th percentile): {q3}")
print(f"Maximum: {maximum}")