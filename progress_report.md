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

### ensembles

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

components to drop from final system: ppmd (confirmed bad on fiction, negative meta weight), semantic_struct (weak, negative meta weight).

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

### hardest cases (updated from literature)
1. paraphrase models (gemini-pro-paraphrase, gpt-4-turbo-paraphrase) — vocabulary features fail, only structural/discourse survive
2. gpt-4.5-preview — attenuated vocabulary fingerprint, concise, emotional warmth markers
3. o3-mini — **reversed signals**: no markdown, minimal hedging, STEM vocabulary → will be misclassified as human by stylometry
4. mixtral-8x7b — 65% of texts unattributable by standard ensemble (arXiv:2503.01659)
5. deepseek-r1 — misclassified as OpenAI; <think> block stripping hides primary fingerprint
6. gemini-2.0-flash — concise by default, harder on short samples

---

## literature findings (pan 2024 top teams)

**tavan & najafi (rank 1, mean 0.924):** binoculars + two lora-finetuned llms (mistral-7b + llama-2). on clean test: 0.995. difference from us: they actually fine-tune end-to-end. our perplexity/tfidf are static. their models learn specific vocabulary fingerprints of each ai model in the dataset.

**huang et al. (rank 2, mean 0.921):** fine-tuned bert + 3-sentence chunk scoring + multi-scale positive-unlabeled (mpu) loss. two innovations we lack: chunk-level inference (prevents ai signal dilution in long docs) and pu loss (handles class imbalance mathematically identically to our 38/62 split problem).

**lorenz et al. (rank 3, mean 0.886):** tfidf top-1000 svm. our tfidf beats theirs in absolute performance, but they ranked 3rd overall vs 9th on clean test → their simpler model was more robust to unicode obfuscation and german text. reveals a generalization blind spot.

**abburi et al. (rank 6 → f1 0.994 at aaai 2025):** roberta + stylometry + e5 concatenated. most similar to our architecture. they add grammatical error count and stop word ratio, which we lack.

---

## what needs to be added to features.py

### already implemented (in models/features.py)
- stylometric_ftrs: n_chars, n_wrds, n_snts, avg_wrd_ln, avg_snt_ln, ttr, mttr, hapax_rate, burstiness (sentence-level), bigram_uniq, verb_ratio, lowercase_ratio, punct_dens, spelling_err_rate
- gltr features (7): frac_top10/100/1000/tail, mean_rank, median_rank, rank_std

### researched but NOT yet implemented (model_research/INDEX.md)
- think-block detector: presence/length of `<think>...</think>` — single strongest fingerprint for deepseek-r1
- boxed answer: `\boxed{` presence (deepseek math outputs)
- sycophantic openers: "Absolutely!", "Certainly!", "Of course!", "Great question!" — llama-2, gpt-3.5
- safety preamble: "As an AI language model,", "I should note that,", "It's important to emphasize"
- rlhf vocab overuse: delve, tapestry, underscore, meticulous, commendable, intricate, testament, realm, pivotal, robust, vibrant, crucial, comprehensive, seamless, leverage, utilize, synergy (~30 words, 10x-200x frequency in ai)
- downtoner rate: somewhat, slightly, rather, fairly, quite — high in llama-2, near-zero in llama-3 (documented difference, arXiv:2510.05136)
- em dash frequency: gpt-4o/gpt-4-turbo specific fingerprint ("chatgpt hyphen")
- markdown density: fraction of lines with headers/bullets/bold — high for gpt-4o/llama-3, ZERO for o3-mini
- paragraph-level burstiness: variance at paragraph length level (complements sentence-level)
- vocabulary absence: fraction of sentences with ZERO rlhf overuse words (human signal)

### new from literature (not yet in any file)
- stop word ratio — abburi et al. (can distinguish register: ai uses fewer stop words in formal writing)
- grammatical error count — abburi et al. (proxy: capitalization/punctuation errors that aren't in our current spelling_err_rate regex)
- stem vocabulary ratio — critical for o3-mini detection: high stem density reverses normal ai signals; need to detect this as a separate cluster
- gemini burstiness band: 0.15-0.22 is gemini-specific range → binary feature: burstiness in [0.15, 0.22]
- multiple gpt-2 sizes for gltr: currently only gpt2 (117m). add gpt2-medium (345m) and gpt2-xl (1.5b) as separate gltr scorers → ensemble of perplexity at different scales gives richer signal

### class imbalance fixes (currently NOT applied)
- class_weight='balanced' in all svm and logistic regression models
- scale_pos_weight = 0.613 in lgbm (= 38/62 = correct ratio to weight human class higher)
- impact: currently ai recall=0.994, ai precision=0.990. balanced weighting will improve human recall at slight ai recall cost — better for robustness

---

## updated project plan

### done ✓
- [x] tfidf baseline eval — acc=0.974 val
- [x] ppmd baseline eval — acc=0.736 val (dropping from ensemble)
- [x] binoculars baseline eval (falcon-7b) — acc=0.865 val
- [x] tfidf retrained (1-4gram, top5000) — mean=0.9894 (best single)
- [x] stylo+gltr classifier (lgbm, 21 features) — mean=0.9699
- [x] e5-small + gbm — mean=0.9554
- [x] semantic struct (kmeans+entropy) — mean=0.7694 (dropping)
- [x] logistic meta-ensemble — acc=0.991, f1=0.993
- [x] per-model analysis (22 models in model_research/)
- [x] genre breakdown analysis

---

### tier 1 — implement now (fast, high impact, no gpu)

**1a. expand features.py with model-specific fingerprints**
priority: critical. effort: 1-2h. expected gain: +1-3% on hard models (mixtral, deepseek, o3-mini).

add to `stylometric_ftrs()`:
- `think_block_present`: bool → float, `think_block_len_ratio`: len(<think>)/total chars
- `boxed_answer`: count of `\boxed{` / n_words
- `sycophancy_rate`: count of opener phrases / n_sents
- `safety_preamble`: count of "As an AI" etc / n_sents
- `rlhf_vocab_rate`: count of ~30 overuse words / n_words
- `downtoner_rate`: count of (somewhat, slightly, rather, fairly, quite) / n_words
- `em_dash_rate`: count of em-dashes (—) / n_chars
- `markdown_density`: lines starting with #, *, - or containing ** / total lines
- `stop_word_ratio`: stop words / n_words
- `para_burstiness`: std(paragraph word counts) / mean(paragraph word counts)
- `stem_vocab_ratio`: fraction of words in curated STEM/scientific word list (for o3-mini)

then retrain stylo+gltr gbm with new features. add class_weight='balanced'.

**1b. fix class weighting**
priority: high. effort: 30min. apply to all existing models on retrain.
- svm/lr: class_weight='balanced'
- lgbm: scale_pos_weight=0.613
- re-evaluate and compare

**1c. tfidf variant — test 3-5gram**
priority: medium. effort: 30min.
- train tfidf with ngram_range=(3,5) min_df=2 as per daigt 2nd place
- test if better than our (1,4) on per-model generalization, not just overall val

---

### tier 2 — medium effort (gpu helpful but not required)

**2a. chunk-level inference (huang et al.)**
priority: high. effort: 2-3h.
- split each text into 3-sentence chunks
- score each chunk independently with stylo+gltr model
- aggregate: mean, max, fraction_above_threshold
- prevents ai signal dilution in long documents (fiction especially)
- especially relevant: deepseek-r1 (longest texts, 866 words avg) may have mixed human/ai paragraphs

**2b. multiple gpt-2 sizes for gltr**
priority: medium. effort: 2-3h.
- add gltr features from gpt2-medium (345m) and gpt2-xl (1.5b)
- our current gltr is only gpt2 (117m)
- ensemble of ranks at different model scales → richer signal
- run on kaggle if local mps is too slow

**2c. e5-large-v2 upgrade**
priority: medium. effort: 1h + compute.
- current e5-small contributes almost nothing to ensemble (+0.18 meta weight)
- multilingual-e5-large or e5-large-v2 expected to be substantially better
- train gbm on e5-large embeddings

**2d. fast-detectgpt on full val**
priority: medium. effort: kaggle compute.
- already implemented in models/fast_detect_gpt.py
- use mixtral as reference model (recommended for hardest cases)
- score not yet collected for full val set

---

### tier 3 — high effort, highest ceiling (needs gpu)

**3a. fine-tune deberta-v3-large**
priority: high if compute available. ceiling: ~0.993+ mean.
- pan25 winner used fine-tuned transformer
- use ranking loss (pairwise) not binary cross-entropy → better calibration
- train on all 23707 texts, validate on 3589
- add to ensemble

**3b. chunk-level deberta with mpu loss (huang et al. full replication)**
priority: high if 3a is done. effort: 3-5 days.
- 3-sentence chunk scoring + mpu (multi-scale positive-unlabeled) loss
- handles the 38/62 class imbalance problem at the loss level
- this was rank 2 at pan 2024

**3c. lora fine-tune mistral-7b (tavan approach)**
priority: medium. effort: significant compute.
- rank 1 at pan 2024 used binoculars + lora-finetuned mistral-7b + llama-2
- lora fine-tuning teaches the model the specific vocabulary fingerprint of each ai model
- high ceiling but high compute cost

**3d. per-genre models**
priority: medium. effort: 2-3h after tier 1 done.
- fiction needs dedicated treatment (50/50 balanced, stylistically different)
- train separate lgbm for each genre using same features
- route at inference time by detected genre (need genre classifier or include genre as feature)

---

### not worth trying
- ppmd (confirmed: negative meta weight, fails on fiction)
- semantic struct as-is (negative meta weight; expanding to 50 dim could help but low priority)
- bert base (deberta strictly better)
- tfidf n-grams > (3,5) (diminishing returns)
- smote/synthetic oversampling (interpolated tfidf vectors meaningless, synthetic e5 samples don't represent pan26 distribution)
- one-class svm alone (too weak as standalone)

---

## realistic performance estimates

| scenario | estimated mean | notes |
|---|---|---|
| current (logistic meta) | ~0.991 val | but includes tfidf which will degrade |
| current without tfidf | ~0.972 val | honest generalization estimate |
| after tier 1 (new features + weighting) | ~0.975-0.980 | better hard-model coverage |
| after tier 2 (chunk + gltr ensemble) | ~0.980-0.985 | better long-doc, richer perplexity signal |
| after tier 3 (deberta fine-tuned) | ~0.990+ | depends on compute access |
| tavan rank 1 pan 2024 | 0.995 (clean test) | fine-tuned llms, highest ceiling |

**submission deadline: may 7, 2026**

---

## pending (priority order)

- [ ] **tier 1a** — expand features.py with 11 new fingerprint features (think-block, rlhf vocab, downtoner, em dash, markdown, stop word, stem vocab, etc)
- [ ] **tier 1b** — fix class weighting in all models (balanced svm, scale_pos_weight lgbm)
- [ ] **tier 1c** — tfidf 3-5gram variant test
- [ ] **tier 2a** — chunk-level inference for stylo+gltr
- [ ] **tier 2b** — gltr with gpt2-medium and gpt2-xl (kaggle)
- [ ] **tier 2c** — e5-large-v2 embedding upgrade
- [ ] **tier 2d** — fast-detectgpt on full val (kaggle, mixtral as reference)
- [ ] **tier 3a** — deberta-v3-large fine-tuned with ranking loss (kaggle/colab)
- [ ] **tier 3b** — chunk deberta + mpu loss
- [ ] **tier 3d** — per-genre models (after tier 1 done)
