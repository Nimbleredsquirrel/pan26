import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
OUT_DIR = DATA_DIR / "output_tfidf_retrain"
OUT_DIR.mkdir(exist_ok=True)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def metrics(tp, fp, tn, fn):
    ttl = tp + fp + tn + fn
    acc = (tp + tn) / ttl if ttl else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1


def eval_preds(val_recs, pred_map, name, thr=0.5):
    tp = fp = tn = fn = 0
    for r in val_recs:
        p = 1 if pred_map[r["id"]] >= thr else 0
        if r["label"] == 1 and p == 1: tp += 1
        elif r["label"] == 0 and p == 1: fp += 1
        elif r["label"] == 0 and p == 0: tn += 1
        else: fn += 1
    acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
    print(f"  {name:<45} acc={acc:.3f}  f1={f1:.3f}  prec={prec:.3f}  rec={rec:.3f}")
    return pred_map, acc, f1


def eval_by_model(val_recs, pred_map, thr=0.5):
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r["model"]].append(r)
    rows = [(m, len(rs), sum(1 for r in rs if pred_map[r["id"]] >= thr) / len(rs))
            for m, rs in model_recs.items()]
    rows.sort(key=lambda x: x[2])
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for m, n, dr in rows:
        print(f"  {m:<40} {n:>5}  {dr:>12.3f}")


def train_eval(trn_recs, val_recs, ngram_rng, max_ftrs, min_df, name):
    trn_txts = [r["text"] for r in trn_recs]
    trn_labels = [r["label"] for r in trn_recs]
    val_txts = [r["text"] for r in val_recs]

    vec = TfidfVectorizer(ngram_range=ngram_rng, max_features=max_ftrs,
                          min_df=min_df, sublinear_tf=True)
    clf = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)

    print(f"  fitting {name}...")
    X_trn = vec.fit_transform(trn_txts)
    clf.fit(X_trn, trn_labels)

    X_val = vec.transform(val_txts)
    proba = clf.predict_proba(X_val)[:, 1]
    pred_map = {r["id"]: float(p) for r, p in zip(val_recs, proba)}

    return pred_map, vec, clf


def main():
    trn_recs = load(DATA_DIR / "train.jsonl")
    val_recs = load(DATA_DIR / "val.jsonl")

    configs = [
        # (ngram_range, max_features, min_df, name)
        ((1, 4), 1000,  1, "tfidf (1,4) top1000 min_df=1  [original]"),
        ((1, 4), 5000,  1, "tfidf (1,4) top5000 min_df=1"),
        ((3, 5), 5000,  2, "tfidf (3,5) top5000 min_df=2  [daigt 2nd place]"),
        ((3, 5), 10000, 2, "tfidf (3,5) top10000 min_df=2"),
        ((2, 5), 5000,  2, "tfidf (2,5) top5000 min_df=2"),
        ((1, 5), 10000, 2, "tfidf (1,5) top10000 min_df=2"),
    ]

    print("tfidf ngram comparison:\n")
    best_f1 = 0
    best_name = None
    best_pred_map = None

    for ngram_rng, max_ftrs, min_df, name in configs:
        pred_map, vec, clf = train_eval(trn_recs, val_recs, ngram_rng, max_ftrs, min_df, name)
        _, acc, f1 = eval_preds(val_recs, pred_map, name)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pred_map = pred_map

    print(f"best: {best_name}  f1={best_f1:.3f}")

    print("per-model breakdown (best config):")
    eval_by_model(val_recs, best_pred_map)

    with open(OUT_DIR / "tfidf_best.jsonl", "w") as f:
        for id, score in best_pred_map.items():
            f.write(json.dumps({"id": id, "label": score}) + "\n")
    print(f"saved to {OUT_DIR}/tfidf_best.jsonl")


if __name__ == "__main__":
    main()
