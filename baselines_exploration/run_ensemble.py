import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()

PREDS = {
    "tfidf": DATA_DIR / "output_tfidf" / "tfidf.jsonl",
    "tfidf_5k": DATA_DIR / "output_tfidf_retrain" / "tfidf_best.jsonl",
    "ppmd": DATA_DIR / "output_ppmd" / "ppmd.jsonl",
    "binoculars": DATA_DIR / "output_binoculars"  / "binoculars.jsonl",
    "stylo": DATA_DIR / "output_stylo_clf" / "stylo.jsonl",
    "stylo_gltr": DATA_DIR / "output_stylo_clf" / "stylo_gltr.jsonl",
    "semantic": DATA_DIR / "output_semantic" / "semantic_k50_n2.jsonl",
    "e5": DATA_DIR / "output_e5_clf" / "e5_e5-small-v2.jsonl",
}


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


def eval_pred_map(val_recs, pred_map, name, thr=0.5):
    tp = fp = tn = fn = 0
    for r in val_recs:
        p = 1 if pred_map[r["id"]] >= thr else 0
        if r["label"] == 1 and p == 1: tp += 1
        elif r["label"] == 0 and p == 1: fp += 1
        elif r["label"] == 0 and p == 0: tn += 1
        else: fn += 1
    acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
    print(f"  {name:<35} acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    return acc, f1


def eval_by_model(val_recs, pred_map, thr=0.5):
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r["model"]].append(r)
    rows = []
    for mdl, recs in model_recs.items():
        det = sum(1 for r in recs if pred_map[r["id"]] >= thr)
        rows.append((mdl, len(recs), det / len(recs)))
    rows.sort(key=lambda x: x[2])
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for mdl, n, dr in rows:
        print(f"  {mdl:<40} {n:>5}  {dr:>12.3f}")


def main():
    val_recs = load(DATA_DIR / "val.jsonl")
    ids = [r["id"] for r in val_recs]
    truth = {r["id"]: r["label"] for r in val_recs}

    all_preds = {}
    for name, path in PREDS.items():
        if not path.exists():
            print(f"  {name} not found, skipping")
            continue
        recs = load(path)
        all_preds[name] = {r["id"]: r["label"] for r in recs}
        print(f"loaded {name}: {len(recs)} preds")

    names = list(all_preds.keys())
    X = np.array([[all_preds[name].get(id, 0.5) for name in names] for id in ids])
    y = np.array([truth[id] for id in ids])

    print("individual baselines:")
    for name in names:
        pm = {id: all_preds[name][id] for id in ids}
        eval_pred_map(val_recs, pm, name)

    print("ensemble strategies:")

    avg = X.mean(axis=1)
    eval_pred_map(val_recs, dict(zip(ids, avg)), "simple average")

    if "tfidf" in names:
        idx = [i for i, n in enumerate(names) if n != "tfidf"]
        avg_no_tfidf = X[:, idx].mean(axis=1)
        eval_pred_map(val_recs, dict(zip(ids, avg_no_tfidf)), "average (no tfidf)")

    accs = []
    for name in names:
        tp = fp = tn = fn = 0
        for r in val_recs:
            p = 1 if all_preds[name][r["id"]] >= 0.5 else 0
            if r["label"] == 1 and p == 1: tp += 1
            elif r["label"] == 0 and p == 1: fp += 1
            elif r["label"] == 0 and p == 0: tn += 1
            else: fn += 1
        accs.append((tp + tn) / len(val_recs))

    wts = np.array(accs)
    wts = wts / wts.sum()
    w_avg = (X * wts).sum(axis=1)
    eval_pred_map(val_recs, dict(zip(ids, w_avg)), f"weighted average {[f'{n}={w:.2f}' for n,w in zip(names,wts)]}")

    mx = X.max(axis=1)
    eval_pred_map(val_recs, dict(zip(ids, mx)), "max confidence")

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    meta = LogisticRegression(C=1.0, max_iter=1000)
    X_sc = scaler.fit_transform(X)
    meta_proba = cross_val_predict(meta, X_sc, y, cv=skf, method="predict_proba")[:, 1]
    eval_pred_map(val_recs, dict(zip(ids, meta_proba)), "logistic meta (5-fold cv)")

    meta.fit(X_sc, y)
    print(f"  meta-learner weights: {dict(zip(names, meta.coef_[0].round(3)))}")

    print("per-model: best ensemble (weighted avg):")
    eval_by_model(val_recs, dict(zip(ids, w_avg)))

    print("per-model: logistic meta (cv):")
    eval_by_model(val_recs, dict(zip(ids, meta_proba)))


if __name__ == "__main__":
    main()
