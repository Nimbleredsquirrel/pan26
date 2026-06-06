# usage: python run_semantic_struct.py [--clusters 50] [--ngram 2]

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from detectors.semantic_struct import SemanticStructClf

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
OUT_DIR = DATA_DIR / "output_semantic"
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


def eval_preds(val_recs, pred_map, thr=0.5):
    tp = fp = tn = fn = 0
    for r in val_recs:
        p = 1 if pred_map[r["id"]] >= thr else 0
        if r["label"] == 1 and p == 1: tp += 1
        elif r["label"] == 0 and p == 1: fp += 1
        elif r["label"] == 0 and p == 0: tn += 1
        else: fn += 1
    acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
    print(f"  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")


def eval_by_genre(val_recs, pred_map, thr=0.5):
    print("  by genre:")
    for g in ["fiction", "essays", "news"]:
        sub = [r for r in val_recs if r["genre"] == g]
        tp = fp = tn = fn = 0
        for r in sub:
            p = 1 if pred_map[r["id"]] >= thr else 0
            if r["label"] == 1 and p == 1: tp += 1
            elif r["label"] == 0 and p == 1: fp += 1
            elif r["label"] == 0 and p == 0: tn += 1
            else: fn += 1
        acc, _, _, f1 = metrics(tp, fp, tn, fn)
        print(f"  {g:<10} acc={acc:.3f}  f1={f1:.3f}  n={len(sub)}")


def eval_by_model(val_recs, pred_map, thr=0.5):
    print("  ai detection rate by model (worst first):")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=2000)
    args = ap.parse_args()

    trn_recs = load(DATA_DIR / "train.jsonl")
    val_recs = load(DATA_DIR / "val.jsonl")

    clf = SemanticStructClf(
        n_clusters=args.clusters,
        n_gram=args.ngram,
        max_ftrs=args.max_features,
    )
    clf.fit(trn_recs, val_recs)

    print("val results:")
    ids, proba = clf.predict_proba(val_recs)
    pred_map = {id: float(p) for id, p in zip(ids, proba)}
    eval_preds(val_recs, pred_map)
    eval_by_genre(val_recs, pred_map)
    eval_by_model(val_recs, pred_map)

    filename = f"semantic_k{args.clusters}_n{args.ngram}.jsonl"
    with open(OUT_DIR / filename, "w") as f:
        for id, score in pred_map.items():
            f.write(json.dumps({"id": id, "label": score}) + "\n")
    print(f"saved to {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
