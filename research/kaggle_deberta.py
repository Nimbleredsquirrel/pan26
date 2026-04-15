# fine-tune deberta-v3-large for ai text detection — intended for kaggle gpu
# paste as kaggle notebook cells after uploading train.jsonl + val.jsonl
#
# setup cell:
#   !pip install transformers accelerate

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from tqdm import tqdm

MDL_NM = "microsoft/deberta-v3-base"   # swap to deberta-v3-large if gpu allows
MAX_LEN = 512
BATCH_SZ = 8
GRAD_ACCUM = 4   # effective batch = 32
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
USE_RANKING_LOSS = True   # pairwise ranking loss (daigt 1st place insight)

TRN_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/train.jsonl"
VAL_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/val.jsonl"
OUT_FILE = "/kaggle/working/deberta_preds.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class TextDataset(Dataset):
    def __init__(self, recs, tok, max_len):
        self.recs = recs
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok(r["text"], max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "label": torch.tensor(r["label"], dtype=torch.float),
        }


def ranking_loss(logits, labels, margin=1.0):
    ai_idx = (labels == 1).nonzero(as_tuple=True)[0]
    hmn_idx = (labels == 0).nonzero(as_tuple=True)[0]
    if len(ai_idx) == 0 or len(hmn_idx) == 0:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    n = min(len(ai_idx), len(hmn_idx))
    ai_scores = logits[ai_idx[:n]]
    hmn_scores = logits[hmn_idx[:n]]
    rl = torch.clamp(margin - (ai_scores - hmn_scores), min=0).mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return rl + 0.5 * bce


dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")

trn_recs = load(TRN_FILE)
val_recs = load(VAL_FILE)

tok = AutoTokenizer.from_pretrained(MDL_NM)
model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
model.to(dev)

trn_ds = TextDataset(trn_recs, tok, MAX_LEN)
val_ds = TextDataset(val_recs, tok, MAX_LEN)
trn_dl = DataLoader(trn_ds, batch_size=BATCH_SZ, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH_SZ * 2, shuffle=False, num_workers=2)

n_steps = (len(trn_dl) // GRAD_ACCUM) * EPOCHS
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * n_steps), n_steps)

best_f1 = 0
best_preds = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    opt.zero_grad()

    for step, batch in enumerate(tqdm(trn_dl, desc=f"epoch {epoch+1}")):
        ids = batch["input_ids"].to(dev)
        msk = batch["attention_mask"].to(dev)
        labels = batch["label"].to(dev)

        logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)

        if USE_RANKING_LOSS:
            loss = ranking_loss(logits, labels)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        loss = loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

    print(f"  epoch {epoch+1} avg loss: {total_loss/len(trn_dl):.4f}")

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="eval"):
            ids = batch["input_ids"].to(dev)
            msk = batch["attention_mask"].to(dev)
            logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    proba = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (proba >= 0.5).astype(int)
    labels_arr = np.array(all_labels).astype(int)
    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    tn = ((preds == 0) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()
    acc = (tp + tn) / len(labels_arr)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  val: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_preds = [(r["id"], float(p)) for r, p in zip(val_recs, proba)]
        torch.save(model.state_dict(), "/kaggle/working/deberta_best.pt")
        print(f"  saved best model (f1={best_f1:.3f})")

with open(OUT_FILE, "w") as f:
    for id, score in best_preds:
        f.write(json.dumps({"id": id, "label": score}) + "\n")
print(f"saved to {OUT_FILE}")
