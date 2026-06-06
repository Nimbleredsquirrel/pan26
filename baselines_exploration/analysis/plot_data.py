import json
import re
import string
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
OUT_DIR = Path("~/Documents/pan-project/baselines_exploration/analysis/plots").expanduser()
OUT_DIR.mkdir(exist_ok=True)


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
    n_words = len(words)
    n_sents = max(len(sents), 1)
    pnct = sum(1 for c in txt if c in string.punctuation)
    return {
        "n_chars": len(txt),
        "n_wrds": n_words,
        "avg_wrd_ln": np.mean([len(w) for w in words]) if words else 0,
        "avg_snt_ln": n_words / n_sents,
        "ttr": len(set(words)) / n_words if n_words else 0,
        "pnct_dens": pnct / len(txt) if txt else 0,
    }


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {path}")


trn = load(DATA_DIR / "train.jsonl")
val = load(DATA_DIR / "val.jsonl")
recs = trn + val
hmn_recs = [r for r in recs if r["label"] == 0]
ai_recs = [r for r in recs if r["label"] == 1]
gnrs = ["fiction", "essays", "news"]
clrs = {"human": "#4878cf", "ai": "#d65f5f"}


# 1. label + genre distributions

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, rs) in zip(axes[:2], [("train", trn), ("val", val)]):
    ttl = len(rs)
    hmn = sum(1 for r in rs if r["label"] == 0)
    ai = ttl - hmn
    ax.bar(["human", "ai"], [hmn, ai], color=[clrs["human"], clrs["ai"]])
    ax.set_title(name)
    ax.set_ylabel("count")
    for i, v in enumerate([hmn, ai]):
        ax.text(i, v + 50, f"{v/ttl:.1%}", ha="center", fontsize=9)

ax = axes[2]
x = np.arange(len(gnrs))
w = 0.35
for i, (lbl, name) in enumerate([(0, "human"), (1, "ai")]):
    cnts = [sum(1 for r in recs if r["genre"] == g and r["label"] == lbl) for g in gnrs]
    ax.bar(x + i*w, cnts, w, label=name, color=clrs[name])
ax.set_xticks(x + w/2)
ax.set_xticklabels(gnrs)
ax.set_title("genre x label")
ax.legend()

fig.suptitle("label and genre distributions")
save(fig, "01_distributions.png")


# 2. text length distributions

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, key, lbl in zip(axes, ["n_chars", "n_wrds"], ["chars", "words"]):
    for rs, name in [(hmn_recs, "human"), (ai_recs, "ai")]:
        vals = [len(tokenize_words(r["text"])) if key == "n_wrds" else len(r["text"]) for r in rs]
        ax.hist(vals, bins=60, alpha=0.6, label=name, color=clrs[name], density=True)
    ax.set_xlabel(lbl)
    ax.set_ylabel("density")
    ax.set_title(f"text length ({lbl})")
    ax.legend()

save(fig, "02_text_length.png")


# 3. stylometric features: human vs ai

feature_keys = ["avg_wrd_ln", "avg_snt_ln", "ttr", "pnct_dens"]
feature_labels = ["avg word length", "avg sentence length", "type-token ratio", "punctuation density"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, k, lbl in zip(axes, feature_keys, feature_labels):
    for rs, name in [(hmn_recs, "human"), (ai_recs, "ai")]:
        vals = [get_ftrs(r["text"])[k] for r in rs]
        ax.hist(vals, bins=50, alpha=0.6, label=name, color=clrs[name], density=True)
    ax.set_xlabel(lbl)
    ax.set_title(lbl)
    ax.legend()

fig.suptitle("stylometric features: human vs ai")
save(fig, "03_stylometrics.png")


# 4. per-model word count + ttr

mdl_recs = defaultdict(list)
for r in recs:
    mdl_recs[r["model"]].append(r)

mdl_rows = []
for mdl, rs in mdl_recs.items():
    avg_w = np.mean([len(tokenize_words(r["text"])) for r in rs])
    avg_t = np.mean([get_ftrs(r["text"])["ttr"] for r in rs])
    mdl_rows.append((mdl, avg_w, avg_t))
mdl_rows.sort(key=lambda x: x[1])

mdls = [r[0] for r in mdl_rows]
avg_ws = [r[1] for r in mdl_rows]
avg_ts = [r[2] for r in mdl_rows]
clr_map = ["#4878cf" if m == "human" else "#d65f5f" for m in mdls]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

axes[0].barh(mdls, avg_ws, color=clr_map)
axes[0].set_xlabel("avg word count")
axes[0].set_title("avg word count per model")
axes[0].axvline(np.mean([len(tokenize_words(r["text"])) for r in hmn_recs]), color="gray", linestyle="--", alpha=0.5)

axes[1].barh(mdls, avg_ts, color=clr_map)
axes[1].set_xlabel("avg ttr")
axes[1].set_title("avg type-token ratio per model")
axes[1].axvline(np.mean([get_ftrs(r["text"])["ttr"] for r in hmn_recs]), color="gray", linestyle="--", alpha=0.5)

fig.suptitle("per-model stats (blue=human baseline)")
save(fig, "04_per_model.png")


# 5. distinctive bigrams

def get_bg_freq(rs):
    cntr = Counter()
    for r in rs:
        words = tokenize_words(r["text"])
        cntr.update(zip(words, words[1:]))
    return cntr

hmn_bg = get_bg_freq(hmn_recs)
ai_bg = get_bg_freq(ai_recs)
hmn_ttl = sum(hmn_bg.values())
ai_ttl = sum(ai_bg.values())

cands = {bg for bg in ai_bg if ai_bg[bg] >= 50 and hmn_bg[bg] >= 10}
ratios = {bg: (ai_bg[bg]/ai_ttl) / (hmn_bg[bg]/hmn_ttl) for bg in cands}

top_ai = sorted(ratios.items(), key=lambda x: -x[1])[:15]
top_hmn = sorted(ratios.items(), key=lambda x: x[1])[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bgs, vals = zip(*top_ai)
axes[0].barh([" ".join(b) for b in bgs], vals, color=clrs["ai"])
axes[0].set_title("most ai-distinctive bigrams")
axes[0].set_xlabel("freq ratio ai/human")

bgs, vals = zip(*top_hmn)
axes[1].barh([" ".join(b) for b in bgs], [1/v for v in vals], color=clrs["human"])
axes[1].set_title("most human-distinctive bigrams")
axes[1].set_xlabel("freq ratio human/ai")

save(fig, "05_distinctive_bigrams.png")


# 6. paraphrase vs normal ai

para = [r for r in ai_recs if "paraphrase" in r["model"]]
norm_ai = [r for r in ai_recs if "paraphrase" not in r["model"]]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
feature_keys2 = ["n_wrds", "ttr", "avg_wrd_ln"]
feature_labels2 = ["word count", "ttr", "avg word length"]

for ax, k, lbl in zip(axes, feature_keys2, feature_labels2):
    v_para = [get_ftrs(r["text"])[k] for r in para]
    v_norm = [get_ftrs(r["text"])[k] for r in norm_ai]
    v_hmn = [get_ftrs(r["text"])[k] for r in hmn_recs]
    ax.hist(v_hmn, bins=40, alpha=0.5, label="human", color=clrs["human"], density=True)
    ax.hist(v_norm, bins=40, alpha=0.5, label="ai", color=clrs["ai"], density=True)
    ax.hist(v_para, bins=40, alpha=0.5, label="paraphrase", color="#e09c3a", density=True)
    ax.set_title(lbl)
    ax.legend(fontsize=8)

fig.suptitle("paraphrase vs normal ai vs human")
save(fig, "06_paraphrase.png")


# 7. genre x model heatmap

mdl_list = [m for m, _ in sorted(Counter(r["model"] for r in ai_recs).items(), key=lambda x: -x[1])]

heat = np.zeros((len(mdl_list), len(gnrs)))
for i, mdl in enumerate(mdl_list):
    for j, g in enumerate(gnrs):
        heat[i, j] = sum(1 for r in ai_recs if r["model"] == mdl and r["genre"] == g)

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(heat, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(gnrs)))
ax.set_xticklabels(gnrs)
ax.set_yticks(range(len(mdl_list)))
ax.set_yticklabels(mdl_list, fontsize=8)
ax.set_title("ai model x genre (sample count)")
fig.colorbar(im, ax=ax)

for i in range(len(mdl_list)):
    for j in range(len(gnrs)):
        ax.text(j, i, int(heat[i, j]), ha="center", va="center", fontsize=7,
                color="white" if heat[i, j] > heat.max()*0.6 else "black")

save(fig, "07_model_genre_heatmap.png")

print("done")
