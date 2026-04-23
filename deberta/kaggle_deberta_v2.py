import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

MDL_NM = "microsoft/deberta-v3-large"
MAX_LEN = 512
BATCH_SZ = 8
GRAD_ACCUM = 4  # effective batch = 32
EPOCHS = 4
LR = 1e-5
WARMUP_RATIO = 0.06
LABEL_SMOOTH = 0.05
ALPHA_RANK = 0.5
POS_WEIGHT = 0.613  # 38/62: down-weights AI majority, balances gradients
RANK_MARGIN = 1.0

TRN_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/train.jsonl"
VAL_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/val.jsonl"
CKPT_DIR = Path("/kaggle/working")
OUT_FILE = CKPT_DIR / "deberta_v2_preds.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class TextDataset(Dataset):
    def __init__(self, recs, tok, max_len, smooth=0.0):
        self.recs = recs
        self.tok = tok
        self.max_len = max_len
        self.smooth = smooth

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok(
            r["text"], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        lbl = float(r["label"])
        if self.smooth > 0:
            lbl = lbl * (1 - self.smooth) + self.smooth / 2
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "label": torch.tensor(lbl, dtype=torch.float),
        }


def weighted_bce(logits, labels):
    pw = torch.tensor(POS_WEIGHT, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)


def ranking_loss(logits, labels, margin=RANK_MARGIN):
    ai_idx = (labels >= 0.5).nonzero(as_tuple=True)[0]
    hmn_idx = (labels < 0.5).nonzero(as_tuple=True)[0]
    if len(ai_idx) == 0 or len(hmn_idx) == 0:
        return weighted_bce(logits, labels)
    n = min(len(ai_idx), len(hmn_idx))
    ai_sc = logits[ai_idx[:n]]
    hmn_sc = logits[hmn_idx[:n]]
    return F.relu(margin - (ai_sc - hmn_sc)).mean()


def combined_loss(logits, labels):
    bce = weighted_bce(logits, labels)
    rank = ranking_loss(logits, labels)
    return ALPHA_RANK * rank + (1 - ALPHA_RANK) * bce


def pan_c_at_1(true_y, pred_y, thr=0.5):
    n = float(len(pred_y))
    nc, nu = 0.0, 0.0
    for gt, pred in zip(true_y, pred_y):
        if pred == thr:
            nu += 1
        elif (pred > thr) == (gt > thr):
            nc += 1.0
    return (1 / n) * (nc + (nu * nc / max(n - nu, 1)))


def pan_f05u(true_y, pred_y, thr=0.5):
    tp = fp = fn = nu = 0
    for t, p in zip(true_y, pred_y):
        if p == thr:
            nu += 1
        elif p > thr and t == 1:
            tp += 1
        elif p > thr and t != 1:
            fp += 1
        elif t == 1 and p <= thr:
            fn += 1
    denom = 1.25 * tp + 0.25 * (fn + nu) + fp
    return (1.25 * tp) / denom if denom else 0.0


def pan_eval(true_y, pred_y):
    scores = {
        "roc-auc": roc_auc_score(true_y, pred_y),
        "brier": 1 - brier_score_loss(true_y, pred_y),
        "c@1": pan_c_at_1(true_y, pred_y),
        "f1": f1_score(true_y, [1 if p >= 0.5 else 0 for p in pred_y]),
        "f0.5u": pan_f05u(true_y, pred_y),
    }
    scores["mean"] = float(np.mean(list(scores.values())))
    return {k: round(v, 4) for k, v in scores.items()}


def print_pan(sc, prefix=""):
    print(f"{prefix}roc-auc={sc['roc-auc']:.4f}  brier={sc['brier']:.4f}  "
          f"c@1={sc['c@1']:.4f}  f1={sc['f1']:.4f}  f0.5u={sc['f0.5u']:.4f}  "
          f"mean={sc['mean']:.4f}")


def eval_by_genre(val_recs, pred_map):
    for g in ["fiction", "essays", "news"]:
        sub = [r for r in val_recs if r.get("genre") == g]
        if not sub:
            continue
        true_y = [r["label"] for r in sub]
        pred_y = [pred_map[r["id"]] for r in sub]
        sc = pan_eval(true_y, pred_y)
        print_pan(sc, prefix=f"  {g:<8} (n={len(sub):4d})  ")


def eval_by_model(val_recs, pred_map, thr=0.5):
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r.get("model", "unknown")].append(r)
    rows = [(m, len(rs), sum(1 for r in rs if pred_map[r["id"]] >= thr) / len(rs))
            for m, rs in model_recs.items()]
    rows.sort(key=lambda x: x[2])
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for m, n, dr in rows:
        print(f"  {m:<40} {n:>5}  {dr:>12.3f}")


def train():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  model: {MDL_NM}")

    trn_recs = load_jsonl(TRN_FILE)
    val_recs = load_jsonl(VAL_FILE)
    print(f"train: {len(trn_recs)}  val: {len(val_recs)}")

    tok = AutoTokenizer.from_pretrained(MDL_NM)
    model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
    model.to(dev)

    trn_ds = TextDataset(trn_recs, tok, MAX_LEN, smooth=LABEL_SMOOTH)
    val_ds = TextDataset(val_recs, tok, MAX_LEN, smooth=0.0)
    trn_dl = DataLoader(trn_ds, batch_size=BATCH_SZ, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SZ * 2, shuffle=False, num_workers=2, pin_memory=True)

    n_steps = (len(trn_dl) // GRAD_ACCUM) * EPOCHS
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * n_steps), n_steps)
    scaler = GradScaler()

    best_mean = 0.0
    best_preds = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(tqdm(trn_dl, desc=f"epoch {epoch+1}/{EPOCHS}")):
            ids = batch["input_ids"].to(dev)
            msk = batch["attention_mask"].to(dev)
            lbls = batch["label"].to(dev)

            with autocast():
                logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                loss = combined_loss(logits, lbls) / GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()

        print(f"  train loss: {epoch_loss / len(trn_dl):.4f}")

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val"):
                ids = batch["input_ids"].to(dev)
                msk = batch["attention_mask"].to(dev)
                with autocast():
                    logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                all_logits.extend(logits.float().cpu().numpy())
                all_labels.extend(batch["label"].numpy())

        proba = torch.sigmoid(torch.tensor(all_logits)).numpy()
        true_y = np.array(all_labels).astype(int)
        sc = pan_eval(true_y.tolist(), proba.tolist())
        print_pan(sc, prefix=f"  epoch {epoch+1} val  ")

        if sc["mean"] > best_mean:
            best_mean = sc["mean"]
            best_preds = [(r["id"], float(p)) for r, p in zip(val_recs, proba)]
            torch.save(model.state_dict(), CKPT_DIR / "deberta_v2_best.pt")
            print(f"  *** new best: mean={best_mean:.4f} → saved checkpoint ***")

    print("\nfinal val breakdown (best checkpoint):")
    pred_map = dict(best_preds)
    true_y = [r["label"] for r in val_recs]
    pred_y = [pred_map[r["id"]] for r in val_recs]
    sc = pan_eval(true_y, pred_y)
    print_pan(sc, prefix="  overall  ")
    print("  by genre:")
    eval_by_genre(val_recs, pred_map)
    print("  ai detection rate by model (worst first):")
    eval_by_model(val_recs, pred_map)

    with open(OUT_FILE, "w") as f:
        for doc_id, score in best_preds:
            f.write(json.dumps({"id": doc_id, "label": score}) + "\n")
    print(f"\nsaved {len(best_preds)} predictions → {OUT_FILE}")


if __name__ == "__main__":
    train()
