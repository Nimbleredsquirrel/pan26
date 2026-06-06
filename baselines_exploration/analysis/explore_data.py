import json
import re
import string
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
TRN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def tokenize_words(txt):
    return re.findall(r"\b\w+\b", txt.lower())


def tokenize_sents(txt):
    return re.split(r"(?<=[.!?])\s+", txt.strip())


def get_ftrs(txt):
    words = tokenize_words(txt)
    sents = tokenize_sents(txt)
    n_chars = len(txt)
    n_words = len(words)
    n_sents = max(len(sents), 1)
    n_uniq = len(set(words))
    pnct = sum(1 for c in txt if c in string.punctuation)

    return {
        "n_chars": n_chars,
        "n_wrds": n_words,
        "n_snts": n_sents,
        "avg_wrd_ln": np.mean([len(w) for w in words]) if words else 0,
        "avg_snt_ln": n_words / n_sents,
        "ttr": n_uniq / n_words if n_words else 0,
        "pnct_dens": pnct / n_chars if n_chars else 0,
    }


def agg_ftrs(recs):
    keys = ["n_chars", "n_wrds", "n_snts", "avg_wrd_ln", "avg_snt_ln", "ttr", "pnct_dens"]
    vals = defaultdict(list)
    for r in recs:
        features = get_ftrs(r["text"])
        for k in keys:
            vals[k].append(features[k])
    return {k: {"mean": np.mean(v), "median": np.median(v), "std": np.std(v)} for k, v in vals.items()}


def print_ftrs(ftrs, indent=2):
    pad = " " * indent
    for k, s in ftrs.items():
        print(f"{pad}{k}: mean={s['mean']:.3f}  median={s['median']:.3f}  std={s['std']:.3f}")


def top_ngrams(recs, n, topk=15):
    cntr = Counter()
    for r in recs:
        words = tokenize_words(r["text"])
        cntr.update(zip(*[words[i:] for i in range(n)]))
    return cntr.most_common(topk)


def print_section(title):
    print(f"{title}:")


trn = load(TRN_FILE)
val = load(VAL_FILE)
all_recs = trn + val

print(f"train: {len(trn)}  val: {len(val)}  total: {len(all_recs)}")


print_section("label distribution")
for split_name, recs in [("train", trn), ("val", val)]:
    ttl = len(recs)
    hmn = sum(1 for r in recs if r["label"] == 0)
    ai = ttl - hmn
    print(f"  {split_name}: human={hmn} ({hmn/ttl:.1%})  ai={ai} ({ai/ttl:.1%})")


print_section("genre distribution")
gnr_cnts = Counter(r["genre"] for r in all_recs)
ttl = len(all_recs)
for g, cnt in gnr_cnts.most_common():
    print(f"  {g}: {cnt} ({cnt/ttl:.1%})")


print_section("genre x label")
for g in gnr_cnts:
    recs_g = [r for r in all_recs if r["genre"] == g]
    hmn = sum(1 for r in recs_g if r["label"] == 0)
    ai = len(recs_g) - hmn
    print(f"  {g}: human={hmn} ({hmn/len(recs_g):.1%})  ai={ai} ({ai/len(recs_g):.1%})")


print_section("ai model distribution (train+val)")
mdl_cnts = Counter(r["model"] for r in all_recs if r["label"] == 1)
for mdl, cnt in mdl_cnts.most_common():
    print(f"  {mdl}: {cnt}")


print_section("text length by label (n_chars)")
for lbl, name in [(0, "human"), (1, "ai")]:
    recs_l = [r for r in all_recs if r["label"] == lbl]
    lens = [len(r["text"]) for r in recs_l]
    print(f"  {name}: mean={np.mean(lens):.0f}  median={np.median(lens):.0f}  "
          f"min={np.min(lens)}  max={np.max(lens)}  std={np.std(lens):.0f}")


print_section("stylometric features: human vs ai")
for lbl, name in [(0, "human"), (1, "ai")]:
    recs_l = [r for r in all_recs if r["label"] == lbl]
    print(f"  {name}:")
    print_ftrs(agg_ftrs(recs_l))


print_section("stylometric features by genre")
for g in gnr_cnts:
    recs_g = [r for r in all_recs if r["genre"] == g]
    print(f"  {g}:")
    print_ftrs(agg_ftrs(recs_g))


print_section("avg word count per model")
mdl_recs = defaultdict(list)
for r in all_recs:
    mdl_recs[r["model"]].append(r)

rows = []
for mdl, recs in mdl_recs.items():
    avg_words = np.mean([len(tokenize_words(r["text"])) for r in recs])
    avg_ttr = np.mean([get_ftrs(r["text"])["ttr"] for r in recs])
    rows.append((mdl, len(recs), avg_words, avg_ttr))

rows.sort(key=lambda x: -x[2])
print(f"  {'model':<40} {'n':>5}  {'avg_wrds':>9}  {'avg_ttr':>8}")
for mdl, n, aw, at in rows:
    print(f"  {mdl:<40} {n:>5}  {aw:>9.1f}  {at:>8.3f}")


print_section("top bigrams: human")
hmn_recs = [r for r in all_recs if r["label"] == 0]
ai_recs = [r for r in all_recs if r["label"] == 1]
for bg, cnt in top_ngrams(hmn_recs, 2):
    print(f"  {' '.join(bg)}: {cnt}")

print_section("top bigrams: ai")
for bg, cnt in top_ngrams(ai_recs, 2):
    print(f"  {' '.join(bg)}: {cnt}")


print_section("most ai-distinctive bigrams (freq ratio ai/human)")
hmn_bg = Counter()
ai_bg = Counter()
for r in hmn_recs:
    words = tokenize_words(r["text"])
    hmn_bg.update(zip(words, words[1:]))
for r in ai_recs:
    words = tokenize_words(r["text"])
    ai_bg.update(zip(words, words[1:]))

hmn_ttl = sum(hmn_bg.values())
ai_ttl = sum(ai_bg.values())

candidates = {bg for bg in ai_bg if ai_bg[bg] >= 50 and hmn_bg[bg] >= 10}
ratios = {bg: (ai_bg[bg]/ai_ttl) / (hmn_bg[bg]/hmn_ttl) for bg in candidates}

print("  ai > human:")
for bg, r in sorted(ratios.items(), key=lambda x: -x[1])[:15]:
    print(f"  {' '.join(bg)}: {r:.2f}x")

print("  human > ai:")
for bg, r in sorted(ratios.items(), key=lambda x: x[1])[:15]:
    print(f"  {' '.join(bg)}: {1/r:.2f}x")


print_section("very short texts (< 200 chars)")
shrt = [r for r in all_recs if len(r["text"]) < 200]
print(f"  count: {len(shrt)}")
for r in shrt[:5]:
    print(f"  [{r['label']} {r['genre']} {r['model']}] {r['text'][:120]!r}")


print_section("paraphrase/obfuscated models")
para_recs = [r for r in all_recs if "paraphrase" in r["model"]]
print(f"  count: {len(para_recs)}")
mdl_cnts = Counter(r["model"] for r in para_recs)
for mdl, cnt in mdl_cnts.most_common():
    print(f"  {mdl}: {cnt}")
