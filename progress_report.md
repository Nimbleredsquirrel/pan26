# progress report: pan26 voight-kampff ai detection

---

## task

binary classification: human-authored (0) vs ai-generated (1) text.
output: confidence score in [0, 1].
dataset: pan25 training data — 23707 train + 3589 val texts, 22 ai models, 3 genres (fiction, essays, news).
test set (pan26): will contain surprise models and unknown obfuscations.

---

## data findings

- label balance: 38% human / 62% ai. genres: fiction 60%, news 20%, essays 20%.
- fiction is the only balanced genre (50/50). essays and news are 80% ai.
- fiction is the hardest genre for all models — human fiction is stylistically consistent (same characters, settings), which looks like ai to most detectors.
- ai writes with longer words (avg 5.0 vs 4.3), higher ttr (0.52 vs 0.47), shorter texts (584 vs 714 words), lower punctuation density.
- ai has very strong cliché bigrams: "a testament to" (164x), "a stark" (107x), "fabric of" (86x), "stark contrast" (78x).
- 670 paraphrase texts (gemini-pro-paraphrase, gpt-4-turbo-paraphrase) — ai texts rewritten to sound human. hardest adversarial case.
- deepseek-r1 is the longest model (866 words avg). paraphrase models are shortest (340-370 words) with highest ttr.

---

## models built and results

official pan metrics: roc-auc, brier (1 - brier_loss), c@1, f1, f0.5u, mean (arithmetic).

### baselines (pan25, pre-trained)

| model | roc-auc | brier | c@1 | f1 | f0.5u | mean |
|---|---|---|---|---|---|---|
| tfidf svm (1-4gram, top1000) | 0.9960 | 0.9510 | 0.9839 | 0.9879 | 0.9813 | 0.9800 |
| binoculars (falcon-7b) | 0.9537 | 0.8848 | 0.8751 | 0.9084 | 0.8818 | 0.9008 |
| ppmd compression | 0.7856 | 0.7987 | 0.7571 | 0.8351 | 0.7778 | 0.7909 |

### our models

| model | roc-auc | brier | c@1 | f1 | f0.5u | mean | notes |
|---|---|---|---|---|---|---|---|
| tfidf retrained (1-4gram, top5000) | 0.9980 | 0.9877 | 0.9855 | 0.9888 | 0.9868 | **0.9894** | best single model |
| stylo + gltr gbm (21 features) | 0.9911 | 0.9673 | 0.9585 | 0.9680 | 0.9646 | 0.9699 | gpt2-based gltr features |
| stylo gbm (14 features) | 0.9872 | 0.9597 | 0.9471 | 0.9591 | 0.9564 | 0.9619 | no gpu, fast |
| e5-small + gbm | 0.9848 | 0.9554 | 0.9384 | 0.9528 | 0.9457 | 0.9554 | semantic embeddings |
| semantic struct (kmeans + entropy) | 0.7635 | 0.8146 | 0.7138 | 0.7846 | 0.7707 | 0.7694 | kogan/gromov approach |

### ensembles (acc/f1 from run_ensemble.py)

| ensemble | acc | f1 | notes |
|---|---|---|---|
| simple average (all 8) | 0.974 | 0.980 | no improvement over tfidf |
| average without tfidf | 0.972 | 0.979 | honest generalization estimate |
| weighted average (all 8) | 0.977 | 0.982 | weights by val accuracy |
| logistic meta (5-fold cv) | **0.991** | **0.993** | best overall |

### per-genre (tfidf_5k, official metrics)

| genre | n | roc-auc | brier | c@1 | f1 | mean |
|---|---|---|---|---|---|---|
| essays | 665 | 0.9998 | 0.9939 | 0.9910 | 0.9944 | 0.9942 |
| fiction | 1837 | 0.9979 | 0.9922 | 0.9918 | 0.9917 | 0.9939 |
| news | 1087 | 0.9941 | 0.9765 | 0.9715 | 0.9824 | 0.9797 |

---

## meta-learner signal analysis

meta-learner weights (logistic regression on all 8 predictions):

```
tfidf_5k    +2.69   dominant, n-gram style patterns
binoculars  +2.24   perplexity ratio, different signal type
tfidf       +1.53   marginal over tfidf_5k
stylo       +0.91   stylometrics
stylo_gltr  +0.67   gltr adds small boost
e5          +0.18   barely contributes (small model)
semantic    -0.49   hurts — noise when better signals present
ppmd        -1.15   hurts — 748 fp overwhelm the signal
```

**core ensemble: tfidf_5k + binoculars + stylo_gltr** — these three provide complementary, non-redundant signals.

---

## feature importance (stylo + gltr model)

top features by lgbm importance:
1. punct_dens — ai uses less punctuation
2. avg_wrd_ln — ai uses longer words
3. hapax_rate — lexical richness measure
4. mttr — moving average type-token ratio
5. lowercase_ratio — ai is more uniformly capitalized
6. gltr_frac_top10 — fraction of tokens in top-10 most likely under gpt-2
7. n_chars — text length
8. bigram_uniq — unique bigram ratio
9. burstiness — variance of sentence lengths (humans more varied)
10. verb_ratio

gltr features (6, 7, 15-19) contribute meaningfully but stylometrics dominate.
spelling_err_rate weakest — our regex proxy is too rough.

---

## per-model analysis

### consistently hardest models (across all approaches)
- **gemini-1.5-pro**: 90-97% detection depending on model — consistently the hardest
- **llama-3.1-8b-instruct**: 85-98% — second hardest
- **gemini-2.0-flash**: 70-99% — hard for compression/perplexity methods

### model-specific failures
- **binoculars** catastrophically fails on gpt-4.5-preview (8.9%) — new architecture breaks falcon-7b perplexity reference
- our stylo+ensemble fully recovers gpt-4.5-preview (95-100%)
- **ppmd** fails on llama-3.1-8b (71%) and gemini-2.0-flash (75%) — their output is less stylistically uniform
- **ppmd** surprisingly strong on paraphrase models (98%) — paraphrased text is more compressible
- **binoculars** degrades on paraphrase (66-84%) — rephrasing breaks perplexity signal

### genre breakdown (logistic meta)
- essays: best (likely because essays are 80% ai — easier to learn)
- fiction: hardest — human fiction is stylistically consistent, looks like ai
- news: middle ground

---

## key conclusions

1. **tfidf is strong but fragile** — 97-99% on val but relies on model-specific n-gram patterns. new models in test set will hurt it.

2. **stylometric features generalize** — burstiness, mttr, bigram uniqueness, hapax rate, gltr_frac_top10 capture general human vs ai writing differences. not tied to specific models.

3. **binoculars and stylometrics are complementary** — binoculars uses probabilistic signal (is this text likely under an llm?), stylometrics uses surface/structural signal (does this text look like human writing?). they fail on different models.

4. **ppmd and semantic structure don't help in ensemble** — both get negative meta weights. their signals are too noisy relative to the others.

5. **e5-small is too weak** — e5-large-v2 or a fine-tuned transformer would likely be much stronger.

6. **realistic generalization estimate: acc≈0.972** — ensemble without tfidf, assuming tfidf degrades on new models.

7. **remaining hard cases**: gemini-1.5-pro and llama-3.1-8b are hard for everyone. likely their writing style is closest to human. worth analyzing specifically.

---

## pending (tier 2-3)

- [ ] fast-detectgpt on full val (kaggle) — zero-shot, per-token probability curvature
- [ ] deberta-v3 fine-tuned with ranking loss (kaggle) — pan25 winner approach
- [ ] add fast-detectgpt + deberta to ensemble — expected to push meta above 0.993
- [ ] e5-large-v2 — likely better than e5-small
- [ ] per-genre models — fiction needs dedicated treatment
