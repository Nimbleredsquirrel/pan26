import json
from pathlib import Path

VAL = Path("~/Documents/pan25-generative-ai-detection-task1-train/val.jsonl").expanduser()
OUT = Path("~/Documents/pan25-generative-ai-detection-task1-train/val_subset50.jsonl").expanduser()

recs = []
with open(VAL) as f:
    for l in f:
        recs.append(json.loads(l))

# 25 human + 25 ai
hmn = [r for r in recs if r["label"] == 0][:25]
ai = [r for r in recs if r["label"] == 1][:25]

with open(OUT, "w") as f:
    for r in hmn + ai:
        f.write(json.dumps(r) + "\n")

print(f"saved {len(hmn)+len(ai)} records to {OUT}")
