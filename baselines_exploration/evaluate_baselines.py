# evaluate all models using official pan metrics:
# roc-auc, brier, c@1, f1, f0.5u, mean

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
VAL_FILE = DATA_DIR / "val.jsonl"

PREDS = {
    "tfidf (pan25)":     DATA_DIR / "output_tfidf"        / "tfidf.jsonl",
    "tfidf_5k (ours)":   DATA_DIR / "output_tfidf_retrain" / "tfidf_best.jsonl",
    "ppmd (pan25)":      DATA_DIR / "output_ppmd"          / "ppmd.jsonl",
    "binoculars (pan25)":DATA_DIR / "output_binoculars"    / "binoculars.jsonl",
    "stylo (ours)":      DATA_DIR / "output_stylo_clf"     / "stylo.jsonl",
    "stylo+gltr (ours)": DATA_DIR / "output_stylo_clf"     / "stylo_gltr.jsonl",
    "semantic (ours)":   DATA_DIR / "output_semantic"      / "semantic_k50_n2.jsonl",
    "e5-small (ours)":   DATA_DIR / "output_e5_clf"        / "e5_e5-small-v2.jsonl",
}


def pan_binarize(y, thr=0.5, triple_valued=False):
    y = np.array(y, dtype=float)
    y = np.ma.fix_invalid(y, fill_value=thr).data
    if triple_valued:
        y[y > thr] = 1
    else:
        y[y >= thr] = 1
    y[y < thr] = 0
    return y


def pan_auc(true_y, pred_y):
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


def pan_brier(true_y, pred_y):
    try:
        return 1 - brier_score_loss(true_y, pred_y)
    except ValueError:
        return 0.0


def pan_c_at_1(true_y, pred_y, thr=0.5):
    n = float(len(pred_y))
    nc, nu = 0.0, 0.0
    for gt, pred in zip(true_y, pred_y):
        if pred == thr:
            nu += 1
        elif (pred > thr) == (gt > thr):
            nc += 1.0
    return (1 / n) * (nc + (nu * nc / n))


def pan_f1(true_y, pred_y, thr=0.5):
    t_filt, p_filt = [], []
    for t, p in zip(true_y, pred_y):
        if p != thr:
            t_filt.append(t)
            p_filt.append(p)
    if not p_filt:
        return 0.0
    p_filt = pan_binarize(p_filt)
    return f1_score(t_filt, p_filt)


def pan_f05u(true_y, pred_y, thr=0.5):
    pred_b = pan_binarize(pred_y, triple_valued=True)
    tp = fp = fn = nu = 0
    for i, pred in enumerate(pred_b):
        if pred == thr:
            nu += 1
        elif pred == 1 and true_y[i] == 1:
            tp += 1
        elif pred == 1 and true_y[i] != 1:
            fp += 1
        elif true_y[i] == 1 and pred != 1:
            fn += 1
    denom = 1.25 * tp + 0.25 * (fn + nu) + fp
    return (1.25 * tp) / denom if denom else 0.0


def pan_eval(true_y, pred_y):
    scores = {
        "roc-auc": pan_auc(true_y, pred_y),
        "brier":   pan_brier(true_y, pred_y),
        "c@1":     pan_c_at_1(true_y, pred_y),
        "f1":      pan_f1(true_y, pred_y),
        "f0.5u":   pan_f05u(true_y, pred_y),
    }
    scores["mean"] = float(np.mean(list(scores.values())))
    return {k: round(v, 4) for k, v in scores.items()}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_preds(path, truth_ids):
    recs = load(path)
    pm = {r["id"]: float(r["label"]) for r in recs}
    return [pm.get(id, 0.5) for id in truth_ids]


def breakdown_by_group(val_recs, pred_map, key, thr=0.5):
    groups = defaultdict(list)
    for r in val_recs:
        groups[r[key]].append(r)
    rows = []
    for grp, recs in sorted(groups.items()):
        true_y = [r["label"] for r in recs]
        pred_y = [pred_map[r["id"]] for r in recs]
        sc = pan_eval(true_y, pred_y)
        rows.append((grp, len(recs), sc))
    return rows


def ai_detect_rate(val_recs, pred_map, thr=0.5):
    model_recs = defaultdict(list)
    for r in val_recs:
        if r["label"] == 1:
            model_recs[r["model"]].append(r)
    rows = [(m, len(rs), sum(1 for r in rs if pred_map[r["id"]] >= thr) / len(rs))
            for m, rs in model_recs.items()]
    return sorted(rows, key=lambda x: x[2])


def main():
    val_recs = load(VAL_FILE)
    truth_ids = [r["id"] for r in val_recs]
    true_y = [r["label"] for r in val_recs]

    print(f"{'model':<25} {'roc-auc':>8} {'brier':>7} {'c@1':>7} {'f1':>7} {'f0.5u':>7} {'mean':>7}")
    all_results = {}
    for name, path in PREDS.items():
        if not path.exists():
            print(f"  {name:<23} [not found]")
            continue
        pred_y = load_preds(path, truth_ids)
        sc = pan_eval(true_y, pred_y)
        all_results[name] = (sc, {id: p for id, p in zip(truth_ids, pred_y)})
        print(f"  {name:<23} {sc['roc-auc']:>8.4f} {sc['brier']:>7.4f} {sc['c@1']:>7.4f} "
              f"{sc['f1']:>7.4f} {sc['f0.5u']:>7.4f} {sc['mean']:>7.4f}")

    best_name = max(all_results, key=lambda n: all_results[n][0]["mean"])
    best_sc, best_pm = all_results[best_name]
    print(f"per-genre breakdown: {best_name}:")
    print(f"{'genre':<12} {'n':>5} {'roc-auc':>8} {'brier':>7} {'c@1':>7} {'f1':>7} {'mean':>7}")
    for grp, n, sc in breakdown_by_group(val_recs, best_pm, "genre"):
        print(f"  {grp:<10} {n:>5} {sc['roc-auc']:>8.4f} {sc['brier']:>7.4f} "
              f"{sc['c@1']:>7.4f} {sc['f1']:>7.4f} {sc['mean']:>7.4f}")

    print(f"ai detection rate by model, {best_name} (worst first):")
    print(f"  {'model':<40} {'n':>5}  {'detect_rate':>12}")
    for m, n, dr in ai_detect_rate(val_recs, best_pm):
        print(f"  {m:<40} {n:>5}  {dr:>12.3f}")

    out = {name: sc for name, (sc, _) in all_results.items()}
    out_path = Path("~/Documents/pan-project/baselines_exploration/results/eval_results.json").expanduser()
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
