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

MDL_NM = "microsoft/deberta-v3-base"
MAX_LEN = 512
BATCH_SZ = 8
GRAD_ACCUM = 4  # effective batch = 32
EPOCHS_SHARED = 2
EPOCHS_GENRE = 2
LR_SHARED = 2e-5
LR_GENRE = 5e-6  # lower LR: don't overwrite shared knowledge
WARMUP_RATIO = 0.06
LABEL_SMOOTH = 0.05

GENRES = ["fiction", "essays", "news"]

TRN_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/train.jsonl"
VAL_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/val.jsonl"
CKPT_DIR = Path("/kaggle/working")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def filter_genre(recs, genre):
    return [r for r in recs if r.get("genre") == genre]


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


def genre_pos_weight(recs):
    n_pos = sum(1 for r in recs if r["label"] == 1)
    n_neg = len(recs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos


def combined_loss(logits, labels, pos_weight, margin=1.0, alpha=0.5):
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
    ai_idx = (labels >= 0.5).nonzero(as_tuple=True)[0]
    hmn_idx = (labels < 0.5).nonzero(as_tuple=True)[0]
    if len(ai_idx) > 0 and len(hmn_idx) > 0:
        n = min(len(ai_idx), len(hmn_idx))
        rank_l = F.relu(margin - (logits[ai_idx[:n]] - logits[hmn_idx[:n]])).mean()
        return alpha * rank_l + (1 - alpha) * bce
    return bce


def pan_c_at_1(true_y, pred_y, thr=0.5):
    n = float(len(pred_y))
    nc = nu = 0.0
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
    print(f"{prefix}auc={sc['roc-auc']:.4f} brier={sc['brier']:.4f} "
          f"c@1={sc['c@1']:.4f} f1={sc['f1']:.4f} f0.5u={sc['f0.5u']:.4f} "
          f"mean={sc['mean']:.4f}")


def fine_tune(trn_recs, val_recs, tok, dev, epochs, lr, label, pos_weight,
              model=None, ckpt_path=None):
    print(f"\n{'='*55}")
    print(f"training: {label}  n_train={len(trn_recs)}  n_val={len(val_recs)}")
    ai_pct = 100 * sum(1 for r in trn_recs if r["label"] == 1) / max(len(trn_recs), 1)
    print(f"  class balance: {ai_pct:.0f}% AI / {100-ai_pct:.0f}% human  pos_weight={pos_weight:.3f}")

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
    model.to(dev)

    trn_ds = TextDataset(trn_recs, tok, MAX_LEN, smooth=LABEL_SMOOTH)
    val_ds = TextDataset(val_recs, tok, MAX_LEN, smooth=0.0)
    trn_dl = DataLoader(trn_ds, batch_size=BATCH_SZ, shuffle=True,
                        num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SZ * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    n_steps = (len(trn_dl) // GRAD_ACCUM) * epochs
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * n_steps), n_steps)
    scaler = GradScaler()

    best_mean = 0.0
    best_preds = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(tqdm(trn_dl, desc=f"{label} ep{epoch+1}")):
            ids = batch["input_ids"].to(dev)
            msk = batch["attention_mask"].to(dev)
            lbls = batch["label"].to(dev)

            with autocast():
                logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                loss = combined_loss(logits, lbls, pos_weight) / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()

        print(f"  ep {epoch+1} loss: {total_loss/len(trn_dl):.4f}")

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(dev)
                msk = batch["attention_mask"].to(dev)
                with autocast():
                    logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                all_logits.extend(logits.float().cpu().numpy())
                all_labels.extend(batch["label"].numpy())

        proba = torch.sigmoid(torch.tensor(all_logits)).numpy()
        true_y = np.array(all_labels).astype(int)
        sc = pan_eval(true_y.tolist(), proba.tolist())
        print_pan(sc, prefix=f"  ep {epoch+1} val ")

        if sc["mean"] > best_mean:
            best_mean = sc["mean"]
            best_preds = [(r["id"], float(p)) for r, p in zip(val_recs, proba)]
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
                print(f"  *** new best: {best_mean:.4f} → {ckpt_path.name} ***")

    return model, best_preds, best_mean


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  model: {MDL_NM}")

    trn_recs = load_jsonl(TRN_FILE)
    val_recs = load_jsonl(VAL_FILE)
    print(f"train: {len(trn_recs)}  val: {len(val_recs)}")
    for g in GENRES:
        n_trn = sum(1 for r in trn_recs if r.get("genre") == g)
        n_val = sum(1 for r in val_recs if r.get("genre") == g)
        print(f"  {g:<8}  train={n_trn}  val={n_val}")

    tok = AutoTokenizer.from_pretrained(MDL_NM)

    shared_pw = genre_pos_weight(trn_recs)
    shared_model, shared_preds, shared_best = fine_tune(
        trn_recs, val_recs, tok, dev,
        epochs=EPOCHS_SHARED, lr=LR_SHARED,
        label="shared_base",
        pos_weight=shared_pw,
        ckpt_path=CKPT_DIR / "shared_base.pt"
    )

    print("\nshared base — val overall:")
    pred_map_shared = dict(shared_preds)
    sc = pan_eval(
        [r["label"] for r in val_recs],
        [pred_map_shared[r["id"]] for r in val_recs]
    )
    print_pan(sc, prefix="  ")
    print("  per genre:")
    for g in GENRES:
        sub = filter_genre(val_recs, g)
        if sub:
            sc_g = pan_eval(
                [r["label"] for r in sub],
                [pred_map_shared[r["id"]] for r in sub]
            )
            print_pan(sc_g, prefix=f"  {g:<8} (n={len(sub):4d})  ")

    genre_preds = {}
    genre_scores = {}

    for g in GENRES:
        trn_g = filter_genre(trn_recs, g)
        val_g = filter_genre(val_recs, g)
        if not trn_g or not val_g:
            print(f"\nskipping {g}: no data")
            continue

        g_model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
        g_model.load_state_dict(torch.load(CKPT_DIR / "shared_base.pt"))

        pw_g = genre_pos_weight(trn_g)
        _, preds_g, best_g = fine_tune(
            trn_g, val_g, tok, dev,
            epochs=EPOCHS_GENRE, lr=LR_GENRE,
            label=f"genre_{g}",
            pos_weight=pw_g,
            model=g_model,
            ckpt_path=CKPT_DIR / f"genre_{g}.pt"
        )

        genre_preds[g] = dict(preds_g)
        genre_scores[g] = best_g
        del g_model
        torch.cuda.empty_cache()

    print("\n" + "="*55)
    print("combined routing evaluation:")

    routed_preds = {}
    for r in val_recs:
        g = r.get("genre", "")
        if g in genre_preds and r["id"] in genre_preds[g]:
            routed_preds[r["id"]] = genre_preds[g][r["id"]]
        else:
            routed_preds[r["id"]] = pred_map_shared[r["id"]]

    true_y = [r["label"] for r in val_recs]
    pred_y = [routed_preds[r["id"]] for r in val_recs]
    sc_routed = pan_eval(true_y, pred_y)
    print_pan(sc_routed, prefix="  routed   ")
    print_pan(pan_eval(
        [r["label"] for r in val_recs],
        [pred_map_shared[r["id"]] for r in val_recs]
    ), prefix="  shared   ")

    print("\nrouted — per genre:")
    for g in GENRES:
        sub = filter_genre(val_recs, g)
        if not sub:
            continue
        sc_g = pan_eval(
            [r["label"] for r in sub],
            [routed_preds[r["id"]] for r in sub]
        )
        src = "genre-specific" if g in genre_preds else "shared fallback"
        print_pan(sc_g, prefix=f"  {g:<8} (n={len(sub):4d}) [{src}]  ")

    print("\nrouted — AI detection rate by model (worst first):")
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r.get("model", "unknown")].append(r)
    rows = [(m, len(rs), sum(1 for r in rs if routed_preds[r["id"]] >= 0.5) / len(rs))
            for m, rs in model_recs.items()]
    rows.sort(key=lambda x: x[2])
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for m, n, dr in rows:
        print(f"  {m:<40} {n:>5}  {dr:>12.3f}")

    out_file = CKPT_DIR / "genre_routed_preds.jsonl"
    with open(out_file, "w") as f:
        for r in val_recs:
            f.write(json.dumps({"id": r["id"], "label": routed_preds[r["id"]]}) + "\n")
    print(f"\nsaved {len(val_recs)} routed predictions → {out_file}")

    shared_out = CKPT_DIR / "genre_shared_preds.jsonl"
    with open(shared_out, "w") as f:
        for r in val_recs:
            f.write(json.dumps({"id": r["id"], "label": pred_map_shared[r["id"]]}) + "\n")
    print(f"saved {len(val_recs)} shared predictions → {shared_out}")

    print("\nsummary:")
    print(f"  shared base  mean={shared_best:.4f}")
    for g, sc in genre_scores.items():
        print(f"  {g:<8}     mean={sc:.4f}")
    print(f"  routed       mean={sc_routed['mean']:.4f}")


if __name__ == "__main__":
    main()
