# usage: python run_fast_detect.py [--calibrate] [--model gpt2-xl] [--subset N]

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detectors.fast_detect_gpt import FastDetectGPT

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
OUT_DIR = DATA_DIR / "output_fast_detect"
TRN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
CALIB_CACHE = Path("~/Documents/pan-project/models/fast_detect_calib.json").expanduser()


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def eval_preds(truth, preds, name):
    tp = fp = tn = fn = 0
    for id, lbl in truth.items():
        if id not in preds:
            continue
        p = 1 if preds[id] >= 0.5 else 0
        if lbl == 1 and p == 1: tp += 1
        elif lbl == 0 and p == 1: fp += 1
        elif lbl == 0 and p == 0: tn += 1
        else: fn += 1
    ttl = tp + fp + tn + fn
    acc = (tp + tn) / ttl if ttl else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"{name}:")
    print(f"  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=100)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    offset, scale = 0.0, 1.0
    if CALIB_CACHE.exists() and not args.calibrate:
        calib = json.loads(CALIB_CACHE.read_text())
        if calib.get("model") == args.model:
            offset = calib["offset"]
            scale = calib["scale"]
            print(f"loaded calibration: scale={scale:.4f}  offset={offset:.4f}")

    det = FastDetectGPT(mdl_nm=args.model, n_smpl=args.n_samples, offset=offset, scale=scale)

    if args.calibrate:
        scale, offset = det.calibrate(TRN_FILE)
        CALIB_CACHE.write_text(json.dumps({"model": args.model, "scale": scale, "offset": offset}))

    val_recs = load(VAL_FILE)
    if args.subset:
        val_recs = val_recs[:args.subset]

    truth = {r["id"]: r["label"] for r in val_recs}

    print(f"scoring {len(val_recs)} texts...")
    preds = {}
    out_path = OUT_DIR / "fast_detect.jsonl"

    with open(out_path, "w") as f_out:
        for i, r in enumerate(val_recs):
            score = det.score(r["text"])
            preds[r["id"]] = score
            f_out.write(json.dumps({"id": r["id"], "label": score}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(val_recs)}")

    eval_preds(truth, preds, f"fast-detectgpt ({args.model})")

    hmn_scores = [preds[r["id"]] for r in val_recs if r["label"] == 0]
    ai_scores = [preds[r["id"]] for r in val_recs if r["label"] == 1]
    print(f"  human scores: mean={np.mean(hmn_scores):.3f}  std={np.std(hmn_scores):.3f}")
    print(f"  ai scores:    mean={np.mean(ai_scores):.3f}  std={np.std(ai_scores):.3f}")


if __name__ == "__main__":
    main()
