# usage:
#   python run_stylo_clf.py               # stylo only
#   python run_stylo_clf.py --gltr        # stylo + gltr (needs gpt2 download)

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detectors.stylo_clf import StyloClf, load

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
MDL_DIR = Path("~/Documents/pan-project/models").expanduser()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def eval_preds(val_recs, ids, proba, thr=0.5):
    pred_map = dict(zip(ids, proba))
    tp = fp = tn = fn = 0
    for r in val_recs:
        p = 1 if pred_map[r["id"]] >= thr else 0
        lbl = r["label"]
        if lbl == 1 and p == 1: tp += 1
        elif lbl == 0 and p == 1: fp += 1
        elif lbl == 0 and p == 0: tn += 1
        else: fn += 1
    ttl = tp + fp + tn + fn
    acc = (tp + tn) / ttl
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")
    return pred_map


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
        ttl = tp + fp + tn + fn
        acc = (tp + tn) / ttl if ttl else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {g:<10} acc={acc:.3f}  f1={f1:.3f}  n={ttl}")


def eval_by_model(val_recs, pred_map, thr=0.5):
    print("  ai detection rate by model (worst first):")
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r["model"]].append(r)
    rows = []
    for model, recs in model_recs.items():
        detected = sum(1 for r in recs if pred_map[r["id"]] >= thr)
        rows.append((model, len(recs), detected / len(recs)))
    rows.sort(key=lambda x: x[2])
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for model, n, dr in rows:
        print(f"  {model:<40} {n:>5}  {dr:>12.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gltr", action="store_true")
    ap.add_argument("--gltr-model", default="gpt2")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    trn_recs = load_jsonl(DATA_DIR / "train.jsonl")
    val_recs = load_jsonl(DATA_DIR / "val.jsonl")

    clf = StyloClf(use_gltr=args.gltr, gltr_mdl=args.gltr_model)
    clf.fit(trn_recs, val_recs)

    print("val results:")
    ids, proba = clf.predict_proba(val_recs)
    pred_map = eval_preds(val_recs, ids, proba)
    eval_by_genre(val_recs, pred_map)
    eval_by_model(val_recs, pred_map)

    print("feature importance (top 20):")
    for name, imp in clf.feature_importance()[:20]:
        bar = "█" * int(imp / max(v for _, v in clf.feature_importance()) * 30)
        print(f"  {name:<25} {imp:>6}  {bar}")

    out_dir = DATA_DIR / "output_stylo_clf"
    out_dir.mkdir(exist_ok=True)
    filename = "stylo_gltr.jsonl" if args.gltr else "stylo.jsonl"
    with open(out_dir / filename, "w") as f:
        for id, score in zip(ids, proba):
            f.write(json.dumps({"id": id, "label": float(score)}) + "\n")
    print(f"saved predictions to {out_dir / filename}")

    imp_path = Path("~/Documents/pan-project/baselines_exploration/results").expanduser()
    imp_name = "feature_importance_gltr.json" if args.gltr else "feature_importance_stylo.json"
    imp = {name: int(v) for name, v in clf.feature_importance()}
    with open(imp_path / imp_name, "w") as f:
        import json as _json
        _json.dump(imp, f, indent=2)
    print(f"saved feature importances to {imp_path / imp_name}")

    import json as _json
    res = {
        "run": "stylo_gltr" if args.gltr else "stylo_only",
        "acc": round(float(sum(1 for r in val_recs if (pred_map[r["id"]] >= 0.5) == bool(r["label"])) / len(val_recs)), 4),
        "features": "stylometric+gltr" if args.gltr else "stylometric",
        "n_features": len(clf.feature_names),
        "model": "lgbm",
        "best_iter": clf.clf.best_iteration_,
    }
    res_path = imp_path / "results.jsonl"
    with open(res_path, "a") as f:
        f.write(_json.dumps(res) + "\n")

    if args.save:
        save_path = MDL_DIR / ("stylo_gltr_clf.pkl" if args.gltr else "stylo_clf.pkl")
        clf.save(save_path)
        print(f"saved model to {save_path}")


if __name__ == "__main__":
    main()
