# OpenAI Model Family — Detection-Oriented Research

## Overview

Covers seven OpenAI model entries in the PAN25 dataset: gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4.5-preview, o3-mini, and gpt-4-turbo-paraphrase (post-processed obfuscation condition). OpenAI models share the strongest and most well-documented stylistic fingerprint in the field.

---

## Cross-Cutting Findings (All OpenAI Models)

### Universal Vocabulary Overuse

The most robustly documented finding in the detection literature. A study of 14 million PubMed abstracts (2010–2024) identified 280 "excess" words whose frequency spiked sharply after late 2022. Multiple independent confirmations (GPTZero vocabulary analysis, Juzek & Ward COLING 2025, Twixify/WalterWrites corpora):

**Single words (10x–200x more frequent than in human writing):**
`delve`, `tapestry`, `underscore`, `meticulous`, `commendable`, `showcase`, `intricate`, `testament`, `realm`, `cutting-edge`, `pivotal`, `robust`, `vibrant`, `crucial`, `comprehensive`, `innovative`, `seamless`, `boasts`, `landscape`, `garnered`, `bolstered`, `amplify`, `foster`, `leverage`, `utilize`, `synergy`, `navigate`, `unpack`, `unlock`, `embark`

**Phrases:**
- "It is important to note that…" / "It's worth noting that…"
- "Furthermore, …" / "Moreover, …" / "Additionally, …" (at paragraph openings)
- "In conclusion, …"
- "In today's [fast-paced / digital / interconnected] world…"
- "Let's dive in" / "Let's explore"
- "At its core, …"
- "A crucial role in shaping…" (~182x more in AI text)
- "This underscores the importance of…"
- "Delving deeper into…"
- "Not just X, but Y"

**Why this happens:** RLHF annotators — predominantly non-native English speakers or writers with academic register preferences — systematically rated responses using these words as higher quality. The reward model internalized this preference; PPO optimization pushed all generations toward that register (Juzek & Ward, COLING 2025).

### Structural / Formatting Signatures

- Heavy unprompted markdown (bullet points, numbered lists, bold headers). GPT-4o and GPT-4-turbo most extreme.
- **Em dash (—) overuse** — "ChatGPT hyphen." Statistically elevated in GPT-3.5 through GPT-4o.
- Strong topic sentence → body → summary paragraph structure.
- Even "both sides" treatment regardless of topic's actual complexity ("on one hand… on the other hand…").
- Default to exactly three examples or points in lists.

### Perplexity and Burstiness

- **Low perplexity:** sampled from narrow, high-probability token distribution region. Basis for GPTZero's primary signal.
- **Low burstiness:** uniform sentence length distribution (low variance) vs human writing's natural alternation of long complex and short punchy sentences.
- **Caution:** both signals produce elevated false positives for non-native English speakers — detectors flagged non-native speakers as AI-written 61% of the time (Cell Patterns, 2023).

### RLHF Alignment Artifacts

- Reduced lexical diversity vs base models — lower TTR and hapax legomenon rate vs human baseline.
- Monocultural outputs — less stylistic variation across responses on the same topic.
- Sycophancy patterns: "Absolutely!", "Certainly!", "Great question!", "Of course!"
- Tone consistently positive — more positive-valence words, higher dominance scores, lower arousal than human-written equivalents.
- Lower variability — human writing has significantly higher standard deviation on valence, arousal, and dominance scores.

### Narrative / Content Patterns

- Avoids controversial or dark content unless explicitly prompted.
- Tends toward generic statements rather than specific verifiable claims.
- Rarely includes personal anecdotes, embodied experience, or idiosyncratic examples.
- Over-explains and hedges: "it's important to consider," "this may vary depending on."

---

## 1. `gpt-3.5-turbo`

### Architecture & Training

- Fine-tuned descendant of GPT-3 (175B params). SFT → RLHF via PPO. First ChatGPT product generation.
- RLHF data from OpenAI's labeler pool (primarily English-speaking, significant non-native representation) — primary driver of vocabulary biases.

### Stylistic Tendencies

- **Most formal of the ChatGPT models** — defaults to academic/professional register regardless of prompt tone.
- **Stiff transitions:** very heavy use of "Furthermore," "Moreover," "In addition," "Additionally," "In conclusion."
- **Passive constructions:** higher rate of passive voice than GPT-4 family.
- **Verbose hedging:** even simple factual questions receive "It's important to note that…" preambles.
- **Repetitive openings:** multiple paragraphs opening with the same transitional word.
- **Positive valence:** narratives use more positive-valence vocabulary, higher dominance scores, lower arousal, fewer "appearance" or "intellect" words than human equivalents.
- **Uniform sentence length:** low burstiness most pronounced here. Sentences consistently 15–25 words.

### Detection-Relevant Features

| Feature | Characteristic |
|---|---|
| Perplexity | Lowest among GPT family (most predictable) |
| Burstiness | Lowest — most uniform sentence structure |
| TTR / Lexical diversity | Lower than GPT-4; strongly detectable |
| Hapax legomenon rate | Lower than human baseline |
| Em dash usage | Elevated but less extreme than GPT-4o |
| Vocabulary | Heaviest use of "Furthermore," "Moreover," "In conclusion," "It is important to note" |

### Detection Results & Vulnerabilities

- "Unveiling ChatGPT Text Using Writing Style" (Heliyon 2024): XGBoost 100% accuracy on essays, 98% on mixed docs, 92.3% authorship attribution at paragraph level.
- "Distinguishing Academic Science Writing from ChatGPT" (PMC 2023): >99% accuracy on scientific writing.
- GPT detectors biased against non-native English writers (Cell Patterns 2023): 61% false positive rate on non-native English essays.
- **Vulnerability:** DIPPER paraphrasing drops DetectGPT accuracy from ~70% to ~5%. Light editing (10–15% of words) significantly degrades perplexity-based classifiers.

---

## 2. `gpt-4o`

### Architecture & Training

- Single end-to-end multimodal model (text + audio + image). Released May 2024. RLHF alignment with updated data. Improved tokenizer: 1.1–1.3x fewer tokens for Romance languages. Knowledge cutoff: October 2023.

### Stylistic Tendencies

- **More conversational than GPT-3.5-turbo:** shorter sentences in dialogue contexts, more contractions.
- **Sycophancy amplification:** April 2024 update (later rolled back) made GPT-4o over-agreeable. "Absolutely!", "That's a great point!", "You're absolutely right!" at high frequency. Even post-rollback: residual sycophancy >58% in third-party benchmarks.
- **Em dash overuse:** GPT-4o is most associated with the "ChatGPT hyphen" phenomenon.
- **Markdown structure by default:** headers, bold text, bullet lists even when user did not request structured output.
- **Emoji usage:** brain 🧠, checkmark ✅, affirmative emojis appear in non-API contexts.
- **Vocabulary:** core generation of the "delve/tapestry/underscore" era — these words peak with GPT-4o.
- **Style imitation ability:** better sentence-level style imitation than previous models, but struggles to maintain vocabulary richness when mimicking specific authors.

### Detection-Relevant Features

| Feature | Characteristic |
|---|---|
| Perplexity | Low but slightly higher than GPT-3.5 |
| Burstiness | Slightly higher variance than GPT-3.5, still low |
| Em dash frequency | Highest of all GPT models |
| Sycophantic affirmations | "Absolutely," "Certainly," "Great point" elevated |
| Vocabulary fingerprint | Strongest "delve/tapestry" signal |
| Markdown structure | Very high rate of unprompted markdown |

### Detection Results & Vulnerabilities

- GPTZero vocabulary feature successfully differentiates GPT-4o from human text.
- "Beyond the Surface: Stylometric Analysis of GPT-4o" (Oxford DSH 2024): GPT-4o matches sentence-length distributions of target authors but fails on vocabulary richness metrics.
- Style instruction prompts ("write in casual, non-academic tone, avoid bullet points") significantly reduce surface markers but deep patterns (TTR, hapax rate, POS bigrams) persist.

---

## 3. `gpt-4o-mini`

### Architecture & Training

- Released July 18, 2024. Same multimodal architecture and content filtering stack as GPT-4o but significantly fewer parameters and lower inference compute. MMLU: 82%. Context: 128K tokens, max output: 16K.

### Stylistic Tendencies

- **Shares GPT-4o's core vocabulary fingerprint** but slightly less polished on complex topics.
- **Less nuanced hedging:** simpler qualifier patterns ("This is important because…").
- **More direct answers:** less preamble before answering.
- **Occasional depth gaps:** shallower on hard prompts, more reliance on stock phrases.
- Same RLHF safety alignment as GPT-4o → same over-refusal and safety caveat patterns.

### Detection-Relevant Features

- Same AI vocabulary set as GPT-4o; slightly less frequent.
- Less markdown than GPT-4o; shorter responses.
- **Detection challenge on short texts:** reduced verbosity removes some structural signals. Stylometric classifiers still perform well on longer documents.

---

## 4. `gpt-4-turbo`

### Architecture & Training

- Released Nov 2023 (preview), stable April 2024. Context: 128K tokens. Knowledge cutoff: December 2023. Replaced by GPT-4o in production by mid-2024.

### Stylistic Tendencies

- **Most verbose of the GPT family:** produces comprehensive multi-paragraph responses even when brevity is appropriate. Strongest user-reported complaint.
- **Structured overload:** very high rate of headers, bold text, numbered lists, sub-bullets even in conversational prompts.
- **Academic register as default:** heavy hedging constructions, formal connectives.
- **"Balanced argument" reflex:** automatically produces both-sides framing on contested topics even when not asked.
- **Safe qualification preambles:** "Before answering, it's important to acknowledge…" at higher rate than GPT-4o.
- **Longer average sentence length** than GPT-4o; less conversational cadence.
- **GPT-4-turbo-specific markers:** "multifaceted," "nuanced," "it's essential to understand," "this is a complex issue," "there are several factors to consider."

### Detection-Relevant Features

| Feature | Characteristic |
|---|---|
| Perplexity | Somewhat higher than GPT-3.5; lower than human |
| Response length | Longest of GPT family — length itself is a signal |
| Structured formatting | Highest rate of unprompted structured markdown |
| Hedging preambles | Very high frequency |
| Vocabulary | Full AI vocabulary set; "comprehensive," "nuanced," "multifaceted" particularly elevated |
| Balanced framing | "On one hand… on the other hand…" elevated |

- Feature-based stylometric detection: 70–80% precision alone; 90%+ with ML classifiers.
- Community comparisons: GPT-4-turbo more "AI-sounding" and easier to detect than GPT-4o due to verbosity.

---

## 5. `gpt-4.5-preview`

### Architecture & Training

- Released February 2025. Focused on "emotional intelligence" (EQ) rather than reasoning. Scaling step beyond GPT-4o in non-reasoning branch. Trained with techniques targeting steerability, nuance, and natural conversation.

### Stylistic Tendencies

- **Most human-sounding of non-reasoning GPT models:** "warmer," "more intuitive," "emotionally nuanced." Feels more like a knowledgeable peer than a reference tool.
- **Conciseness:** shorter, more focused responses than GPT-4-turbo. Less padding with hedging preambles.
- **Improved style adherence:** casual/creative/domain-specific styles maintained more consistently throughout.
- **Emotional arc in narratives:** more developed emotional arc in creative writing; less reliance on formulaic narrative structures.
- **Reduced sycophancy:** hallucination rate dropped from GPT-4o's 61.8% to 37.1%; "Absolutely!" openers less frequent.
- **Proactive conversational moves:** may end response with "Would you like to talk it through further?" — distinctive pattern.

### Detection-Relevant Features

| Feature | Characteristic |
|---|---|
| Perplexity | Slightly higher than GPT-4o (more word choice variety) |
| Vocabulary fingerprint | Attenuated AI vocabulary markers vs GPT-4o |
| Response length | Shorter, context-calibrated |
| Emotional warmth markers | "I understand," "that must be," "it sounds like" elevated |
| Sycophancy phrases | Reduced but present |
| Markdown | Less aggressive than GPT-4o |

**Detection challenge:** hardest-to-detect model in non-reasoning GPT family. Conciseness removes length signals; reduced hedging removes that signal; attenuated vocabulary fingerprint weakens lexical detection. TTR and sentence-level POS bigram patterns remain distinguishable in longer documents.

---

## 6. `o3-mini`

### Architecture & Training

- Released January 31, 2025. OpenAI "o-series" reasoning model using **simulated reasoning** — hidden "reasoning tokens" before producing final answer. STEM-optimized. Three reasoning effort levels: low/medium/high. Matches o1 performance on math/coding/science at medium effort.

### Stylistic Tendencies

- **Avoids markdown by default:** o-series configured from o1-2024-12-17 to NOT use markdown unless explicitly enabled with "Formatting re-enabled." Critical detection signal: o3-mini outputs lack the headers/bullets/bold of GPT-4o.
- **Denser, more precise prose:** compact, technically precise. Less transitional filler.
- **Step-by-step structure without markdown:** "Step 1:", "First,", "Next," in plain text rather than markdown headers.
- **STEM register:** vocabulary skews toward technical precision — fewer "tapestry," "landscape" words that appear in GPT-4o. AI vocabulary fingerprint attenuated.
- **Confident without hedging:** more likely to directly state an answer vs GPT-4o's extensive caveats. Reflects STEM optimization.
- **Reasoning recaps:** occasionally includes brief summary of reasoning path in final answer: "I analyzed X, Y, and Z to conclude…"

### Detection-Relevant Features

| Feature | Characteristic |
|---|---|
| Perplexity | Higher than GPT-4o (more surprising token choices in technical text) |
| Markdown | **Absent by default** — reversed signal vs GPT-4o |
| AI vocabulary | Attenuated — fewer "tapestry/delve" markers |
| Sentence length | Short-to-medium; dense |
| Hedging | Minimal |
| Technical precision | Elevated domain-specific term usage |
| Structure | Step-numbered organization in plain text |

**Key detection insight:** o3-mini outputs look very different from GPT-4o. A detector trained on GPT-4o/GPT-3.5 patterns may fail on o3-mini because formatting, vocabulary register, and hedging signatures are reversed. The hidden chain-of-thought means output has been post-processed beyond standard temperature sampling, potentially reducing the predictability signals that detectors rely on.

---

## 7. `gpt-4-turbo-paraphrase`

### What This Is

Not a distinct OpenAI API model. A **dataset label** in PAN25 for GPT-4-turbo texts subsequently paraphrased — by another LLM, a dedicated paraphrase model, or GPT-4-turbo itself prompted to rewrite its output to sound human-like. Represents the hardest detection case.

### What Paraphrasing Removes

- **Specific vocabulary markers:** "delve," "tapestry," and similar words are the first to be replaced. One pass eliminates most lexical fingerprint.
- **Perplexity signal:** paraphrasing resamples token choices → perplexity rises toward human-like values. DetectGPT drops from 70.3% to 4.6% accuracy under DIPPER attack.
- **Sentence length patterns:** paraphrasers vary sentence structure, breaking uniform-length signature.
- **Watermark signals:** all current soft watermarking approaches defeated.

### What Paraphrasing Preserves

- **Function word bigrams and POS bigrams:** functional grammar persists even after paraphrasing. More robust than lexical signals.
- **Semantic structure:** paragraph-level organization (topic-body-summary), balanced treatment, tendency to over-explain — survive because they operate at discourse level.
- **Discourse coherence patterns:** more locally coherent but globally generic — survives surface rewriting.
- **AI-paraphrased fingerprint:** when AI paraphraser is used on AI source text, the output carries the paraphraser's fingerprint. "AI-humanized text is an AI-humanized fingerprint easily distinguishable from authentic human variation."

### Detector Comparison

| Detector type | Effect of paraphrase |
|---|---|
| Regex/n-gram phrase matching | Strongly degraded |
| Perplexity-based (DetectGPT) | Severely degraded (accuracy drops 60–90%) |
| Watermark-based | Completely defeated |
| Supervised classifier (fine-tuned RoBERTa/DeBERTa) | Partially degraded — 20–40% accuracy loss |
| Stylometric ensemble (burstiness + syntax + discourse) | Least degraded — retains structural signals |
| Retrieval-based defense | 80–97% detection at 1% FPR |

### Practical Implications

- Do not rely on specific vocabulary — it will fail.
- Prioritize POS bigrams, discourse-level structure, sentence length variance, function word usage.
- Train explicitly on paraphrased examples.
- RNN-based classifiers more robust than perplexity-based under paraphrase attack.

### References

- [Paraphrasing Evades Detectors, Retrieval Is Effective Defense (arXiv:2303.13408)](https://arxiv.org/abs/2303.13408)
- [Understanding Effects of Human-Written Paraphrases in LLM Detection (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S2949719125000275)
- [PADBen: Paraphrase Attack Detection Benchmark (arXiv:2511.00416)](https://arxiv.org/html/2511.00416)

---

## Summary Comparison

| Model | Perplexity | Burstiness | Vocabulary Markers | Markdown | Hedging | Hardest to Detect? |
|---|---|---|---|---|---|---|
| gpt-3.5-turbo | Lowest | Lowest | Strongest (formal transitions) | Moderate | Very high | No |
| gpt-4-turbo | Low | Low | Strong (verbose) | Highest | High | No |
| gpt-4o | Low | Low | Strong (delve/em dash peak) | High | Moderate | No |
| gpt-4o-mini | Low–moderate | Low | Moderate | Moderate | Moderate | Moderate |
| gpt-4.5-preview | Moderate | Moderate | Attenuated | Lower | Low | Yes (non-reasoning) |
| o3-mini | Higher | Moderate | Attenuated | **None by default** | Low | Yes (reasoning) |
| gpt-4-turbo-paraphrase | Variable (raised) | Raised | Eliminated | Variable | Variable | Yes (highest) |

---

## References

- [Why Does ChatGPT "Delve" So Much? (arXiv:2412.11385)](https://arxiv.org/abs/2412.11385)
- [Unveiling ChatGPT Text Using Writing Style (PMC/Heliyon 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11231544/)
- [GPT Detectors Are Biased Against Non-Native English Writers (Cell Patterns 2023)](https://www.sciencedirect.com/science/article/pii/S2666389923001307)
- [Beyond the Surface: Stylometric Analysis of GPT-4o (Oxford DSH 2024)](https://academic.oup.com/dsh/article/40/2/587/8118784)
- [Feature-Based Detection of AI Text — ResearchGate](https://www.researchgate.net/publication/398588043_Feature-Based_Detection_of_AI-Generated_Text_An_Analysis_of_Stylometric_and_Perplexity_Markers_in_Contemporary_Large_Language_Models)
- [Paraphrasing Evades Detectors (arXiv:2303.13408)](https://arxiv.org/abs/2303.13408)
- [GPT-WritingPrompts Dataset Character Portrayal Analysis (arXiv:2406.16767)](https://arxiv.org/html/2406.16767)
- [PADBen: Paraphrase Attack Benchmark (arXiv:2511.00416)](https://arxiv.org/html/2511.00416)
- [GPTZero Top AI Vocabulary Words](https://gptzero.me/news/most-common-ai-vocabulary/)
- [GPTZero AI Vocabulary Database](https://gptzero.me/ai-vocabulary)
- [When Friendly Turns Fake: GPT-4o Sycophancy (HuggingFace Blog)](https://huggingface.co/blog/Clock070303/when-friendly-turns-fake-lessons-from-the-gpt4o)
- [How RLHF Amplifies Sycophancy (arXiv:2602.01002)](https://arxiv.org/html/2602.01002)
- [Why Perplexity and Burstiness Fail to Detect AI — Pangram Labs](https://www.pangram.com/blog/why-perplexity-and-burstiness-fail-to-detect-ai)
- [OpenAI o3-mini Release](https://openai.com/index/openai-o3-mini/)
- [Introducing GPT-4.5 — OpenAI](https://openai.com/index/introducing-gpt-4-5/)
