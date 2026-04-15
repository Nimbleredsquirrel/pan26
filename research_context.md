# research context: pan26 voight-kampff ai detection

---

## our own prior work (start here)

### gromov, dang, kogan, yerbolova (2024) — "spot the bot: the inverse problems of nlp"
*peerj computer science, vol. 10*

**approach:**
- represent semantic space of text as vectors of n-grams (tf-idf/svd or word2vec/cbow)
- use wishart clustering to find coarse-grained structure of semantic space
- bigram clusters form around "main" words — structurally different for bots vs humans
- use topological data analysis (tda) to find "holes" in semantic space
- extract entropy-complexity features from cluster structure
- train svm / dt / rf on these features

**key finding:**
- statistically significant differences exist in coarse-grained structure of human vs bot semantic spaces
- svm with all features: f1=0.98 (english), 0.82 (russian), 0.63 (german), 0.82 (french), 0.74 (vietnamese)

**critical methodological contribution (directly relevant to pan26):**
- novel problem statement: split bots into train/test subsets
- model trained on some bots, tested on *other* bots it has never seen
- this is exactly the pan26 setup (surprise models in test set)
- validates that coarse-grained semantic structure generalizes across unseen models

**features used:**
- n-gram cluster assignments (wishart)
- entropy of cluster distribution
- complexity measures over semantic space
- network-based features (concept co-occurrence)

**ref:** https://peerj.com/articles/cs-2152/

---

### gromov, kogan (2023) — "spot the bot: coarse-grained partition of semantic paths"
*pattern recognition and machine intelligence, springer, pp. 348–355*

- earlier version of the above
- establishes that semantic path structure differs between bots and humans
- uses t-sne visualization to show separation of human/bot/lstm/gpt2 clusters

---

## detection methods — zero-shot (no labeled data needed)

### binoculars — hans et al. (2024)
*arxiv 2401.12070*

**how it works:**
- uses two closely related llms: observer (base model) and performer (instruct model)
- score = -log_perplexity(performer) / cross_perplexity(observer, performer)
  = -(log p_perf(x)) / H(p_obs, p_perf)
- ai text: both models assign high probability → high score
- human text: models disagree more → lower score

**pan25 baseline defaults:** falcon-7b (observer) + falcon-7b-instruct (performer)
**pan25 offset:** PAN25_ACCURACY_OFFSET = 1.4617 (calibrated on training set)

**results:**
- >90% detection of chatgpt at 0.01% fpr (zero-shot)
- pan25 val: acc=0.865, f1=0.901
- weakness: gpt-4.5-preview only 8.9% detection rate — new models break it
- paraphrase degrades it: gpt-4-turbo-paraphrase only 66.1% detection

**why it fails on new models:** falcon-7b was trained before many modern models existed;
its probability estimates are less reliable for text from architecturally different models

---

### fast-detectgpt — bao et al. (2023)
*arxiv 2310.05130, iclr 2024*

**how it works:**
- builds on detectgpt (mitchell et al. 2023) but 340x faster
- key insight: ai text sits in high-probability regions of the llm landscape
- instead of sampling text perturbations (slow), samples alternative tokens at each position
- score per position t: (log p(x_t | x_{<t}) - mean_k[log p(x̃_t^k | x_{<t})]) / std_k[...]
- overall score: mean over all positions
- higher score → text more probable than random → ai-generated

**vs detectgpt:**
- detectgpt: perturb full text with t5, compare log probs → ~800 forward passes per text
- fast-detectgpt: sample tokens at each position → 1 forward pass + sampling
- 75% better auroc than detectgpt, 340x faster

**our implementation:** models/fast_detect_gpt.py
- uses gpt2-xl as reference model by default
- calibrated offset+scale via logistic regression on training set

---

### detectgpt — mitchell et al. (2023)
*icml 2023, arxiv 2301.11305*

- ai text sits in negative curvature regions of log p(x)
- perturb text with t5 → compare log prob of original vs perturbations
- score = log p(x) - mean[log p(x̃)] for perturbations x̃
- improved from 0.81 → 0.95 auroc on gpt-neox fake news detection
- slow (needs many perturbations), fast-detectgpt replaces this

---

### detectllm — su et al. (2023)
*arxiv 2306.05540*

- two variants: detectllm-lrr (fast) and detectllm-npr (accurate)
- uses log-rank of actual tokens rather than raw log-probability
- lrr: rank of token in sorted probability distribution
- npr: log-rank normalized by perturbation-based estimate
- gains of +1.75 (lrr) to +3.9 (npr) auroc over detectgpt

---

## detection methods — supervised

### tfidf + svm (pan25 baseline)
- tf-idf vectorizer: 1-4 ngrams, top 1000 features
- linear svc trained on pan25 training set
- pan25 val: acc=0.974, f1=0.980
- **warning:** overfits to known model patterns — will degrade on new models
- weakest on gemini-1.5-pro (93.8%) — stylistically closest to human writing

### roberta / deberta fine-tuned (pan25 top systems)
- pan25 winner (mdok): mean=0.989, roc-auc=0.995
- most top systems used fine-tuned transformer classifiers
- modernbert-based system: mean=0.928, roc-auc=0.962
- graph-based system (valdez-valenzuela): mean=0.929 — uses text structure graphs

---

## detection methods — compression-based

### ppmd compression cosine (pan25 baseline)
- split text in half, compress each half and full text with ppmd
- score = (c(x) + c(y) - c(xy)) / sqrt(c(x)*c(y))
- high score → stylistically uniform → likely ai
- pan25 val: acc=0.736, f1=0.817
- strong on paraphrase models (98.4%) — paraphrased text is MORE compressible
- weak on fiction (66.6%) — human fiction is stylistically consistent

### sculley & brodley (2006), halvani et al. (2017)
- foundational compression-as-similarity papers
- compression size as proxy for statistical predictability
- basis for ppmd detector

---

## adversarial / limits of detection

### sadasivan et al. (2023) — reliability of ai text detection
- theoretical argument: detection becomes unreliable when text is heavily paraphrased
- paraphrasing can reduce detectgpt accuracy from 70.3% to 4.6%
- retrieval-based defense (database of api outputs) partially counters this
- **implication for pan26:** paraphrase models (gemini-pro-paraphrase, gpt-4-turbo-paraphrase)
  represent exactly this attack — our system must handle them

### ippolito et al. (2020) — acl
- texts harder to detect by classifiers are also harder for humans to detect
- model quality and detectability are inversely correlated
- better models → harder to detect → this gets worse over time

### jakesch, hancock, naaman (2023) — pnas
- humans use specific heuristics: formal language, structured writing, hedging phrases
- "a testament to", "stark contrast", "highlights the" — classic ai clichés
- our bigram analysis confirms this: these are 50-160x more frequent in ai

---

## competition-specific findings (pan 2024, 2025)

### pan 2024 results
- winner: mean=0.924, roc-auc=0.961
- binoculars (falcon-7b): mean=0.741 — much weaker than pan25 binoculars
- top systems beat baselines significantly → fine-tuned transformers dominate

### pan 2025 results
- winner (mdok): mean=0.989, roc-auc=0.995 — near-perfect
- 2nd place (graph-based, valdez-valenzuela): mean=0.929
- 3rd place (modernbert): mean=0.928
- baselines: tfidf mean≈0.978, binoculars mean≈0.877, ppmd mean≈0.786

### what worked in 2025:
1. fine-tuned transformer (roberta/deberta/modernbert) — best overall
2. graph-based text structure features — novel, 2nd place
3. ensemble of features — consistent approach

---

## our data analysis findings

### dataset (pan25, used for pan26 development)
- 27296 total: 23707 train + 3589 val
- label balance: ~38% human, ~62% ai
- genres: fiction 60.6%, news 19.9%, essays 19.5%
- fiction is only balanced genre (50/50); essays+news are 80% ai

### key stylometric differences (human vs ai)
- avg word length: ai=5.0 vs human=4.3 — ai writes more formally
- type-token ratio: ai=0.52 vs human=0.47 — ai avoids repetition (counterintuitive)
- text length: human longer (714 words vs 584) — ai is more concise
- punctuation density: human=0.034 vs ai=0.027 — human uses more punctuation (dialogue)

### most ai-distinctive bigrams (ratio >50x vs human)
"a testament to", "a stark", "fabric of", "stark contrast", "confines of",
"reflecting the", "a reminder", "emphasizing the", "highlights the"

### hardest cases
- fiction: hardest genre for all baselines (human fiction = stylistically consistent)
- gpt-4.5-preview: 8.9% detection by binoculars (new architecture)
- paraphrase models: binoculars degrades to 66-84%, ppmd remains strong

### per-model insights
- deepseek-r1 writes the most (866 words avg) — verbose reasoning model
- paraphrase models write least (340-370 words) and have highest ttr
- llama-2 variants: lower ttr, easier to detect
- gpt-4.5-preview: highest ttr (0.60), hardest to detect

---

## ensemble systems — recent competition work

### arxiv 2505.11550 — roberta + e5 + stylometrics ensemble
*pan/semeval-style competition, 5th place binary (f1=0.994), 1st multiclass (f1=0.627)*

**three architectures tried:**

1. **full:** roberta-base detector + bilstm on token-level perplexity + e5 embeddings → fc layer
2. **optimized:** roberta-base + stylometric features + e5 embeddings → fc layer (best binary: f1=0.994)
3. **simple:** e5 embeddings + 11 stylometric features → gradient boosting (best multiclass: f1=0.627)

**key finding:** replacing token-level perplexity features with stylometric features improved performance.
simpler is better — bilstm over perplexity adds complexity without gain.

**stylometric features used (11):**
- unique word count
- stop word count
- moving average type-token ratio (mttr) ← better than static ttr
- hapax legomenon rate (words appearing exactly once) ← new, we don't have this
- word count
- bigram uniqueness ← ratio of unique bigrams to total bigrams
- sentence count
- average sentence length
- lowercase letter ratio ← proxy for capitalization style
- burstiness ← variance of sentence lengths (high = human, low = ai)
- verb ratio ← verbs / total words

**what failed:** roberta binary detector doesn't extend to multiclass (0.19 f1).
perfect val f1 (1.0) but much lower test — classic overfitting to seen models.

**e5 embeddings:** multilingual-e5 or e5-large, mean pooling over tokens.
captures semantic similarity patterns that stylometrics miss.

**gradient boosting + e5 + stylometrics:**
- interpretable, fast, no gpu needed for inference
- surprisingly competitive — doesn't need roberta at all for decent results
- this is a strong candidate for our approach

**critical insight for pan26:** the binary roberta model overfits to known models.
on test set with new models, stylometric + embedding features generalize better.
mirrors our finding that tfidf (97.4% val) will likely degrade on test.

**taxonomy from this paper:**
- statistical: gltr (entropy + confidence scores over token distribution)
- zero-shot: fast-detectgpt, binoculars
- fine-tuned: roberta/deberta/modernbert
- feature-based: ngrams, liwc, readability, stylometry
- ensemble: any combination of above

---

## kaggle daigt competition — "llm detect ai generated text" (2024)
*vanderbilt university + the learning agency lab, $110k prize, 1097 teams*

**setup differences from pan26:**
- task: student essays only (persuade 2.0 corpus), not multi-genre
- training set: 1375 human essays + only 3 ai essays (near-useless)
- hidden test: essays from 7 prompts, generated by unknown llms
- pan26 is harder: multi-genre, 22 known models + surprise models, obfuscation

---

### 1st place — raja biswas (rbiswasfc)
*github: https://github.com/rbiswasfc/llm-detect-ai*

**three-component ensemble:**

1. **fine-tuned llms (qlora)** — instruction-tuned llms fine-tuned for binary detection
   - trained on essays generated by 9 llm families: tinyllama, pythia, bloom, gpt-2, opt, falcon, mpt, llama-13b, mistral
   - key: generate training data from many diverse llms → robust to unseen models

2. **deberta-v3-large with ranking loss**
   - pairwise ranking loss instead of binary cross-entropy
   - ranks ai essays higher in suspicion than human essays
   - better than classification loss because it captures relative signals

3. **embedding model + knn**
   - trained with supervised contrastive loss
   - knn similarity to training examples as inference feature

**critical insight: typo injection**
- human student essays have spelling errors; ai essays don't
- models without typo injection use spelling as a shortcut — fails on test
- solution: inject typos into ai training data OR correct typos in human data

**hardware:** 4x a100 40gb or a6000 48gb

---

### 2nd place — guanshuo xu
**approach:** tf-idf (n-grams 3-5) + linear classifiers + gradient boosted trees
- trigram–5-gram tf-idf with `min_df=2` → +0.02 auc boost over default settings
- sgd classifier + multinomial naive bayes + catboost/lightgbm ensemble
- no neural nets — pure feature engineering beat most fine-tuned models

---

### 5th place — linguistic ninjas
- trained on 1.7 million essays — largest training set
- domain adaptation to align training with hidden test distribution

---

### 6th place — chg0901
*github: https://github.com/chg0901/6th-kaggle-DAIGT-entropy-based-text-detector*
- **entropy-based anomaly detection**: per-token entropy distributions
- **one-class svm** trained only on human text — ai text = anomaly
- human writing: irregular entropy; ai writing: uniform, peaked at most likely token
- top-10 without fine-tuning any classification neural network

---

### 8th place — abdullah meda
**features:**
- **perplexity (ppl)**: ai text has low perplexity under the generating llm
- **gltr features**: giant language model test room
  - each word gets a rank under a lm (how likely was this word?)
  - ai text clusters in top-k most likely words
  - human text is spread across more of the distribution
  - extract: fraction of words in top-10, top-100, top-1000, tail

---

## cross-cutting insights from daigt (directly applicable to pan26)

### what worked
1. **tf-idf trigrams–5-grams (n=3,5) + `min_df=2`**: single most effective simple feature
   - captures stylistic phrase patterns; ai uses longer, more uniform phrases
   - outperforms word2vec alone; simpler and better than svd reduction

2. **deberta-v3-large**: best transformer backbone consistently; outperforms roberta, bert, smaller variants

3. **diverse llm training data**: teams that generated training data from many different llm families
   generalized better to unseen test models — directly mirrors pan26 surprise model challenge

4. **ranking loss on deberta**: better than binary cross-entropy for relative ranking tasks

5. **ensemble diversity**: tfidf-linear + deberta + tree model + knn = complementary signals

6. **gltr features**: fraction of tokens in top-10/100/1000 most likely tokens under a lm —
   fast to compute, strong zero-shot signal, complementary to raw perplexity

7. **one-class svm on entropy**: unsupervised, generalizes to unseen models (6th place!)

### what didn't work
- prompt id as model input: overfits to topic, not style
- selectkbest, truncatedsvd, essay-quality features: no improvement over raw tfidf
- bert base: much worse than deberta (~0.74 vs 0.96+ auc)
- n-grams larger than (3,5): diminishing returns
- single model without ensemble: bigger leaderboard shake-up on private set
- training on only original 3 ai samples: collapsed to predicting everything human

### daigt → pan26 translation
| daigt finding | pan26 implication |
|---|---|
| diverse llm training data → generalization | our train set already has 22 models — good |
| tf-idf 3-5gram best simple feature | our tfidf uses 1-4gram — try 3-5gram variant |
| typo injection matters | fiction human texts may have more irregularities — worth analyzing |
| ranking loss > classification loss | try ranking loss on deberta if we fine-tune |
| gltr features strong | implement gltr: fraction top-10/100/1000 tokens under gpt2 |
| one-class svm on entropy | interesting fallback for unseen model robustness |
| deberta-v3-large best backbone | priority transformer if we go neural |
| public → private lb shake-up common | don't trust val accuracy alone — test per-model generalization |

---

## gltr features (from 8th place daigt + original gltr paper)

for each token in text, query a reference lm for its rank (position in sorted probability list):
- `frac_top10`: fraction of tokens with rank ≤ 10
- `frac_top100`: fraction of tokens with rank ≤ 100
- `frac_top1000`: fraction of tokens with rank ≤ 1000
- `frac_tail`: fraction of tokens with rank > 1000
- ai text: most tokens cluster in top-10/100 (model writes what's most likely)
- human text: more tokens in tail (humans make surprising word choices)

can use gpt-2 (or any causal lm) as reference — fast, no gpu needed for gpt-2 small

---

## our current approach plan

### implemented
- [x] tfidf baseline (pan25) — acc=0.974 val
- [x] ppmd baseline (pan25) — acc=0.736 val
- [x] binoculars baseline (falcon-7b, pan25) — acc=0.865 val
- [x] fast-detectgpt (models/fast_detect_gpt.py) — not yet evaluated on full val

### next steps (priority order)

**tier 1 — fast to implement, strong signal:**
- [ ] stylometric classifier (gbm) with full feature set:
      existing: n_chars, n_wrds, avg_wrd_ln, avg_snt_ln, ttr, pnct_dens
      new: burstiness, mttr, hapax rate, bigram uniqueness, verb ratio, lowercase ratio
- [ ] gltr features: frac_top10/100/1000/tail under gpt-2 (fast, no gpu)
- [ ] tf-idf retrained with n-grams (3,5) — test if better than (1,4)
- [ ] e5 embeddings + gradient boosting

**tier 2 — medium effort:**
- [ ] run fast-detectgpt on full val set (kaggle)
- [ ] ensemble: stylometrics + gltr + fast-detectgpt + tfidf
- [ ] per-genre models (fiction is hard)

**tier 3 — high effort, high ceiling:**
- [ ] fine-tuned deberta-v3-large (best pan25 approach, needs gpu)
- [ ] deberta with ranking loss instead of classification loss
- [ ] coarse-grained semantic structure features (kogan/gromov: wishart + entropy-complexity)

**not worth trying:**
- one-class svm alone (too weak)
- bert base (deberta strictly better)
- tfidf n-grams > (3,5) (diminishing returns)

### key design constraints
- each text processed in isolation (no cross-sample info)
- must run in docker without external api calls
- output: confidence score in [0, 1]
- submission deadline: may 7, 2026
