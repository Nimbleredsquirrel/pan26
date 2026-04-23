import json
import re
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
MAX_LEN = 256
BATCH_SZ = 16
GRAD_ACCUM = 2  # effective batch = 32
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.06
CHUNK_SIZES = [1, 3, 5]
PRIOR = 0.62  # P(AI) in training data
AGG_METHODS = ["max", "mean", "p90"]

TRN_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/train.jsonl"
VAL_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/val.jsonl"
CKPT_DIR = Path("/kaggle/working")

_ABBREVS = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K|approx|est)\.",
    re.IGNORECASE
)
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[{])')


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = _ABBREVS.sub(lambda m: m.group().replace(".", "<<DOT>>"), text)
    sents = _SENT_BOUNDARY.split(text)
    sents = [s.replace("<<DOT>>", ".").strip() for s in sents]
    return [s for s in sents if len(s) > 8]


def text_to_chunks(text, chunk_size, stride=None):
    if stride is None:
        stride = max(1, chunk_size // 2)
    sents = split_sentences(text)
    if not sents:
        return [text]
    chunks = []
    for i in range(0, max(1, len(sents) - chunk_size + 1), stride):
        chunk = " ".join(sents[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    if not chunks:
        chunks = [text[:512]]
    return chunks


def build_chunk_records(recs, chunk_size):
    chunks = []
    for r in recs:
        for chunk in text_to_chunks(r["text"], chunk_size):
            chunks.append({
                "doc_id": r["id"],
                "text": chunk,
                "label": r["label"],
                "genre": r.get("genre", ""),
            })
    return chunks


class ChunkDataset(Dataset):
    def __init__(self, chunks, tok, max_len):
        self.chunks = chunks
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        c = self.chunks[i]
        enc = self.tok(
            c["text"], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "label": torch.tensor(float(c["label"]), dtype=torch.float),
            "doc_id": c["doc_id"],
        }


def chunk_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "doc_id": [b["doc_id"] for b in batch],
    }


def nnpu_loss(logits, labels, prior=PRIOR):
    """Non-negative PU loss (Kiryo et al., NeurIPS 2017).

    Human texts (label=0) treated as unlabeled — may contain AI passages.
    L_nnPU = π * R+(f) + max(0, R_U-(f) - π * R+-(f))
    """
    ai_mask = labels >= 0.5
    hmn_mask = ~ai_mask

    if ai_mask.sum() == 0 or hmn_mask.sum() == 0:
        return F.binary_cross_entropy_with_logits(logits, labels)

    ones_ai = torch.ones(ai_mask.sum(), device=logits.device)
    zeros_ai = torch.zeros(ai_mask.sum(), device=logits.device)
    zeros_hm = torch.zeros(hmn_mask.sum(), device=logits.device)

    r_pos = F.binary_cross_entropy_with_logits(logits[ai_mask], ones_ai, reduction="mean")
    r_unl = F.binary_cross_entropy_with_logits(logits[hmn_mask], zeros_hm, reduction="mean")
    r_pn = F.binary_cross_entropy_with_logits(logits[ai_mask], zeros_ai, reduction="mean")

    return prior * r_pos + torch.clamp(r_unl - prior * r_pn, min=0.0)


def aggregate_chunks(chunk_scores_by_doc, method="max"):
    doc_scores = {}
    for doc_id, scores in chunk_scores_by_doc.items():
        s = np.array(scores)
        if method == "max":
            doc_scores[doc_id] = float(s.max())
        elif method == "mean":
            doc_scores[doc_id] = float(s.mean())
        elif method == "p90":
            doc_scores[doc_id] = float(np.percentile(s, 90))
        elif method == "p75":
            doc_scores[doc_id] = float(np.percentile(s, 75))
        else:
            raise ValueError(f"unknown aggregation: {method}")
    return doc_scores


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


def score_chunks(model, dl, dev):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(dev)
            msk = batch["attention_mask"].to(dev)
            with autocast():
                logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
            scores = torch.sigmoid(logits.float()).cpu().numpy()
            all_scores.extend(scores.tolist())
    return all_scores


def train_scale(trn_recs, val_recs, chunk_size, dev, tok, model=None):
    print(f"\n{'='*60}")
    print(f"training: chunk_size={chunk_size}")

    trn_chunks = build_chunk_records(trn_recs, chunk_size)
    val_chunks = build_chunk_records(val_recs, chunk_size)
    print(f"  train chunks: {len(trn_chunks)}  val chunks: {len(val_chunks)}")
    print(f"  avg chunks/doc train: {len(trn_chunks)/len(trn_recs):.1f}")

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
    model.to(dev)

    trn_ds = ChunkDataset(trn_chunks, tok, MAX_LEN)
    val_ds = ChunkDataset(val_chunks, tok, MAX_LEN)
    trn_dl = DataLoader(trn_ds, batch_size=BATCH_SZ, shuffle=True,
                        num_workers=2, pin_memory=True, collate_fn=chunk_collate)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SZ * 2, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=chunk_collate)

    n_steps = (len(trn_dl) // GRAD_ACCUM) * EPOCHS
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * n_steps), n_steps)
    scaler = GradScaler()

    best_mean = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(tqdm(trn_dl, desc=f"epoch {epoch+1}")):
            ids = batch["input_ids"].to(dev)
            msk = batch["attention_mask"].to(dev)
            lbls = batch["label"].to(dev)

            with autocast():
                logits = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                loss = nnpu_loss(logits, lbls) / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()

        print(f"  epoch {epoch+1} loss: {total_loss/len(trn_dl):.4f}")

        chunk_scores = score_chunks(model, val_dl, dev)

        scores_by_doc = defaultdict(list)
        for c, score in zip(val_chunks, chunk_scores):
            scores_by_doc[c["doc_id"]].append(score)

        true_by_doc = {r["id"]: r["label"] for r in val_recs}
        for method in AGG_METHODS:
            doc_scores = aggregate_chunks(scores_by_doc, method=method)
            true_y = [true_by_doc[doc_id] for doc_id in doc_scores]
            pred_y = [doc_scores[doc_id] for doc_id in doc_scores]
            sc = pan_eval(true_y, pred_y)
            print_pan(sc, prefix=f"  epoch {epoch+1} {method:<6} ")

        doc_scores = aggregate_chunks(scores_by_doc, method="max")
        true_y = [true_by_doc[doc_id] for doc_id in doc_scores]
        pred_y = [doc_scores[doc_id] for doc_id in doc_scores]
        sc = pan_eval(true_y, pred_y)
        if sc["mean"] > best_mean:
            best_mean = sc["mean"]
            torch.save(model.state_dict(), CKPT_DIR / f"chunk{chunk_size}_best.pt")
            print(f"  *** new best: {best_mean:.4f} → saved chunk{chunk_size}_best.pt ***")

    return model, best_mean


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  model: {MDL_NM}")

    trn_recs = load_jsonl(TRN_FILE)
    val_recs = load_jsonl(VAL_FILE)
    print(f"train: {len(trn_recs)}  val: {len(val_recs)}")

    tok = AutoTokenizer.from_pretrained(MDL_NM)
    results = {}

    for chunk_size in CHUNK_SIZES:
        _, best_mean = train_scale(trn_recs, val_recs, chunk_size, dev, tok)
        results[chunk_size] = best_mean

    print("\nscale summary:")
    for cs, m in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  chunk_size={cs}  best_mean={m:.4f}")

    best_scale = max(results, key=results.get)
    print(f"\nbest scale: {best_scale}-sentence chunks")

    print(f"\nreloading best {best_scale}-sentence model for final val inference...")
    model = AutoModelForSequenceClassification.from_pretrained(MDL_NM, num_labels=1)
    model.load_state_dict(torch.load(CKPT_DIR / f"chunk{best_scale}_best.pt"))
    model.to(dev)

    val_chunks = build_chunk_records(val_recs, best_scale)
    val_ds = ChunkDataset(val_chunks, tok, MAX_LEN)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SZ * 2, shuffle=False,
                        num_workers=2, collate_fn=chunk_collate)
    chunk_scores = score_chunks(model, val_dl, dev)

    scores_by_doc = defaultdict(list)
    for c, score in zip(val_chunks, chunk_scores):
        scores_by_doc[c["doc_id"]].append(score)

    true_by_doc = {r["id"]: r["label"] for r in val_recs}

    print("\nfinal val — all aggregation methods:")
    best_preds = None
    best_mean = 0.0
    best_method = "max"

    for method in AGG_METHODS + ["mean"]:
        doc_scores = aggregate_chunks(scores_by_doc, method=method)
        true_y = [true_by_doc[doc_id] for doc_id in doc_scores]
        pred_y = [doc_scores[doc_id] for doc_id in doc_scores]
        sc = pan_eval(true_y, pred_y)
        print_pan(sc, prefix=f"  {method:<6} ")
        if sc["mean"] > best_mean:
            best_mean = sc["mean"]
            best_preds = [(doc_id, doc_scores[doc_id]) for doc_id in doc_scores]
            best_method = method

    print(f"\nbest aggregation: {best_method}  mean={best_mean:.4f}")

    print("per-genre (best aggregation):")
    pred_map = dict(best_preds)
    for g in ["fiction", "essays", "news"]:
        sub = [r for r in val_recs if r.get("genre") == g]
        if not sub:
            continue
        true_y = [r["label"] for r in sub]
        pred_y = [pred_map[r["id"]] for r in sub]
        sc = pan_eval(true_y, pred_y)
        print_pan(sc, prefix=f"  {g:<8} (n={len(sub):4d})  ")

    out_file = CKPT_DIR / "chunk_mpu_preds.jsonl"
    with open(out_file, "w") as f:
        for doc_id, score in best_preds:
            f.write(json.dumps({"id": doc_id, "label": score}) + "\n")
    print(f"\nsaved {len(best_preds)} predictions → {out_file}")


if __name__ == "__main__":
    main()
