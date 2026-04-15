# run fast-detectgpt on full val set — intended for kaggle gpu
# paste this as a kaggle notebook cell after uploading val.jsonl as a dataset
#
# setup cell (run first):
#   !pip install transformers accelerate

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

VAL_FILE = "/kaggle/input/pan25-generative-ai-detection-task1-train/val.jsonl"
OUT_FILE = "/kaggle/working/fast_detect_gpt2xl.jsonl"
MDL_NM = "gpt2-xl"
MAX_LEN = 512
N_SMPL = 100


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def raw_score(txt, model, tok, dev, max_len=MAX_LEN, n_smpl=N_SMPL):
    inputs = tok(txt, return_tensors="pt", max_length=max_len,
                 truncation=True).to(dev)
    ids = inputs.input_ids
    if ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(**inputs).logits
    log_p = F.log_softmax(logits[0, :-1, :], dim=-1)
    actual_log_prob = log_p.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
    probs = log_p.exp().clamp(min=1e-9)
    samples = torch.multinomial(probs, n_smpl, replacement=True)
    sample_log_probs = log_p.gather(1, samples)
    mu = sample_log_probs.mean(dim=1)
    sigma = sample_log_probs.std(dim=1).clamp(min=1e-8)
    return ((actual_log_prob - mu) / sigma).mean().item()


def calibrate_sigmoid(raw_scores, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array(raw_scores).reshape(-1, 1)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_sc, labels)
    k = clf.coef_[0][0] / scaler.scale_[0]
    b = clf.intercept_[0] - clf.coef_[0][0] * scaler.mean_[0] / scaler.scale_[0]
    offset = float(-b / k) if k != 0 else 0.0
    return float(k), offset


dev = get_device()
print(f"device: {dev}")

tok = AutoTokenizer.from_pretrained(MDL_NM)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MDL_NM, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
model.to(dev)
model.eval()

val_recs = load(VAL_FILE)
print(f"scoring {len(val_recs)} texts...")

raw_scores = []
for r in tqdm(val_recs):
    raw_scores.append(raw_score(r["text"], model, tok, dev))

# calibrate on val labels (for analysis — in production calibrate on train)
labels = [r["label"] for r in val_recs]
scale, offset = calibrate_sigmoid(raw_scores, labels)
print(f"calibrated: scale={scale:.4f}  offset={offset:.4f}")

proba = [float(torch.sigmoid(torch.tensor(scale * (s - offset))).item()) for s in raw_scores]

with open(OUT_FILE, "w") as f:
    for r, p in zip(val_recs, proba):
        f.write(json.dumps({"id": r["id"], "label": p}) + "\n")

print(f"saved to {OUT_FILE}")

thr = 0.5
tp = fp = tn = fn = 0
for r, p in zip(val_recs, proba):
    pred = 1 if p >= thr else 0
    if r["label"] == 1 and pred == 1: tp += 1
    elif r["label"] == 0 and pred == 1: fp += 1
    elif r["label"] == 0 and pred == 0: tn += 1
    else: fn += 1
ttl = tp + fp + tn + fn
acc = (tp + tn) / ttl
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
print(f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
