# Meta Llama Family — AI Text Detection Research

## Overview

Covers four Meta Llama models in the PAN25 dataset. All share decoder-only autoregressive transformer architecture but differ substantially in post-training alignment method, tokenizer generation, vocabulary, and resulting stylistic fingerprints. Llama 2 Chat (RLHF/PPO) and Llama 3.x Instruct (SFT + rejection sampling + DPO) are detectably different sub-families.

---

## 1. `llama-2-7b-chat`

### Architecture & Training

- Decoder-only, 7B params, 32 layers, no GQA (GQA only at 34B+). Context: 4,096 tokens. Trained on 2T tokens, cutoff Sep 2022.
- **Tokenizer:** SentencePiece BPE, **32,000 tokens** — small by modern standards. Heavy subword splitting of rare/non-English words, more tokens-per-word than Llama 3.
- **Post-training:** SFT (human-annotated pairs) → iterative RLHF with PPO + dual reward models (helpfulness + safety) + Ghost Attention for multi-turn system prompt adherence.
- Optimal sampling temp per paper: T ∈ [1.2, 1.3] — shifts each RLHF iteration.
- Default system prompt baked in: `"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe…"`

### Stylistic Tendencies

- **RLHF over-optimization:** reward models had ~65–75% accuracy on preferences → model over-indexed on annotator-preferred surface features: formal, cautious, hedged, repetitively structured.
- **Sycophancy:** softens disagreement, validates user framing, avoids confrontation.
- **Opener clichés:** "Of course!", "Sure!", "Certainly!", "Absolutely!", "I'd be happy to help with that.", "Great question!"
- **Safety language:** "I want to make sure I'm being responsible…", "I should note that…", "Please be aware that…"
- **Hedge phrases:** "it's worth noting that…", "it's important to remember that…", "while there are different perspectives…"
- **Structural:** 3–5 paragraph responses, explicit transition adverbs — "Furthermore," "Additionally," "Moreover." Restates user's question in multi-turn.
- **Sentence length:** uniform, low burstiness. Alternates short declarative + medium compound. Limited variation.
- **Formality:** always formal, rarely uses contractions or colloquialisms.

### Detection-Relevant Features

- **Perplexity:** low under Llama-2-family scorer; SentencePiece tokenizer makes perplexity lower for Llama-2-generated text.
- **Vocabulary:** restricted lexical diversity vs human news; fewer adjectives, more symbols/numbers (arXiv:2308.09067).
- **Negative emotion words:** *increase* with model size — 7B has fewer than 70B.
- **Burstiness:** consistently low (sentence std dev ~2.0–3.8 words vs 4.8–7.2 for humans).
- **N-gram repetition:** measurable in closing paragraphs where summary phrases recur.
- **Punctuation:** heavy em-dash (—) and colon (:) for examples; parenthetical `(e.g., …)`.
- **Chat template artifacts:** `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>` — hard identifiers if prompt artifacts leak.

### Detection Results & Vulnerabilities

- arXiv:2503.01659: ensemble achieves **precision 0.9988, FPR 0.0004** for Llama vs. other families.
- Llama texts cluster together across model generations — 7B and 70B share detectable overlap.
- **Vulnerability:** prompt-guided paraphrasing significantly reduces detection rates (arXiv:2305.10847). 7B most susceptible due to limited capacity → formulaic outputs.

### References

- [Llama 2 paper (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- [HF: meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Detecting Stylistic Fingerprints (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)
- [Contrasting Linguistic Patterns (arXiv:2308.09067)](https://arxiv.org/abs/2308.09067)
- [Linguistic Characteristics Survey (arXiv:2510.05136)](https://arxiv.org/abs/2510.05136)

---

## 2. `llama-2-70b-chat`

### Architecture & Training

- Same family as 7B but 70B params with **GQA** (applied only at 70B+). Context: 4,096 tokens. Same 2T token training corpus, Sep 2022 cutoff.
- **Tokenizer:** identical SentencePiece BPE 32k — same tokenization artifacts as 7B.
- **Post-training:** same pipeline (SFT → PPO RLHF), but more optimization steps and higher-quality human annotation. Separate annotation regime for larger models.
- 70B showed superior perf in 36% of head-to-head comparisons vs ChatGPT (Llama 2 paper human evals).

### Stylistic Tendencies

- **All Llama-2-7b-chat patterns apply** — more polished versions, not different ones.
- **Longer responses:** 4–6 paragraphs where 7B gives 2–3; more elaborate transition scaffolding.
- **More negative emotion vocabulary** than 7B (size-dependent effect, per arXiv:2510.05136).
- **Structural symmetry:** parallel structure in responses — three-part answers with header/body/mini-conclusion per section. Low structural burstiness at paragraph level.
- **Hedging escalation:** more qualifiers per sentence — "it's generally considered," "in most cases," "while this can vary," "from a certain perspective."
- **High-frequency clichés:**
  - "I hope that helps!" / "I hope this answers your question!"
  - "Let me know if you have any other questions!"
  - "It's worth noting that…" (very high freq)
  - "It's important to emphasize that…"
  - "There are several key factors to consider…"
  - "On one hand… on the other hand…"
  - "In conclusion," / "To summarize," as paragraph transitions (not just endings)
- **False refusals:** more frequent than 7B; stronger safety reward signal → more safety preambles on innocuous requests.
- **Downtoner use:** "somewhat," "slightly," "rather," "fairly," "quite" — extensive.

### Detection-Relevant Features

- **Perplexity:** lower than 7B under Llama-2 scorer (more self-consistent). Low under out-of-family scorer too.
- **Per-token surprisal variance:** lower and more uniform than human text — detectable by PAWN-style detectors.
- **Burstiness:** low (est. sentence std dev 2.5–4.0 words).
- **N-gram fingerprints at 4–6 gram level:** "it is worth noting that," "in order to," "as previously mentioned," "on the other hand," "it is important to."
- **Vocabulary:** formal Latinate — "utilize" over "use," "facilitate" over "help," "demonstrate" over "show," "incorporate" over "include." High freq abstract nouns: -tion/-ity/-ness endings.
- **Punctuation:** perfect grammar, no sentence fragments, no comma splices, no informal dashes, no typos.

### Detection Results & Vulnerabilities

- arXiv:2503.01659: same high-precision Llama family detection. Fingerprint shared across 7B and 70B.
- arXiv:2412.19076 (ALTA 2024 Shared Task): token-probability methods work well due to consistent low-entropy generation.
- **70B slightly harder than 7B** for simple perplexity thresholds — higher coherence and more varied construction. Structural and n-gram features remain discriminative.

### References

- [Llama 2 paper (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- [HF: meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [ALTA 2024 Shared Task (arXiv:2412.19076)](https://arxiv.org/abs/2412.19076)
- [FAID: Fine-Grained AI Detection (arXiv:2505.14271)](https://arxiv.org/abs/2505.14271)
- [PAWN: Not All Tokens Are Equal (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S156625352500538X)

---

## 3. `llama-3.1-8b-instruct`

### Architecture & Training

- Decoder-only, 8B params, GQA on all sizes. Context: **128,000 tokens** (vs 4,096 in Llama 2). Trained on ~15T tokens, cutoff Dec 2023.
- **Tokenizer:** switched to **tiktoken BPE**, vocabulary **128,256 tokens** (4x larger than Llama 2). Fewer multi-token splits of common English words → different tokenization patterns and lower tokens-per-word.
- Special tokens: `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>` + 256 reserved. Chat template uses role headers — structurally distinct from Llama 2's `[INST]/<<SYS>>`.
- **Post-training:** SFT on 25M+ synthetic examples + human-annotated data → rejection sampling → **DPO** (not PPO). DPO avoids separate reward model at inference; produces different alignment artifact patterns than PPO.
- Substantial synthetic training data bootstrapped from stronger Llama 3 variants → stylistic patterns partially reflect learned Llama-family style.

### Stylistic Tendencies

- **Less RLHF over-alignment than Llama 2:** Meta explicitly reports reduced false refusal rates and increased response diversity.
- **Decreased downtoner usage** — documented in arXiv:2510.05136 as a specific measurable POS-level difference from Llama 2. "Somewhat," "slightly," "rather" appear significantly less often.
- **More clausal coordination** — more compound sentences with "and/but/or" rather than complex subordination (ibid.).
- **Markdown bias:** strong tendency for unprompted markdown — bullet points (`-` or `*`), numbered lists, bold headers (`**text**`), code blocks. Reflects large synthetic training dataset and 128k context paradigm.
- **High-frequency word markers:**
  - "Delve" — documented ~48x more frequent in LLM text than human text (Max Planck Institute 2024); especially associated with Llama 3 and GPT-4 fine-tuned outputs
  - "Multifaceted," "comprehensive," "nuanced," "robust," "pivotal," "crucial," "key takeaway"
  - "In today's world" / "In today's fast-paced world"
  - "Moreover," "Furthermore," "Additionally" (present but less reflexive than Llama 2)
  - "Let me know if you have any questions!" / "Feel free to ask if you need more information!"
- **Response scaffold:** definition → numbered/bulleted elaboration → closing summary. Highly consistent three-part structure.
- **Formality:** slightly less formal than Llama 2; more willing to use casual register when prompted casually.

### Detection-Relevant Features

- **Tokenizer fingerprint:** 128k tiktoken vocabulary produces different byte-pair merge patterns than 32k SentencePiece. Token boundary analysis can fingerprint the tokenizer generation.
- **Perplexity:** zero-shot AUROC **0.730–0.751** using Llama 3.1 8B itself as scorer. Drops to 0.503–0.596 with few-shot prompting → perplexity-based detection brittle but structural methods robust.
- **Burstiness:** slightly higher than Llama 2 7B, still below human norms. Markdown lists artificially flatten burstiness (all bullet items roughly equal length).
- **Vocabulary:** fewer adjectives, formal Latinate preferences; tiktoken means different per-token probability profiles for technical content.
- **Structural marker:** "explain X" → one-sentence definition → numbered list → closing summary. Three-part scaffold highly consistent.
- **Special token leakage:** `<|eot_id|>` and role header tokens are hard identifiers if stripping is incomplete.

### Detection Results & Vulnerabilities

- arXiv:2503.01659: Llama family detectable at high precision (same family-level fingerprint as Llama 2).
- **Evasion via style prompt:** IFEval score 82.6 → strong style-mimicry response to prompts like "write in a casual, human-like style." Surface features shift; perplexity/structural signals harder to mask.
- DPO alignment reduces safety-phrase-based features — detectors relying on "as an AI language model" and "I cannot and will not" will have lower recall on Llama 3.x outputs.

### References

- [Llama 3 Herd paper (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [HF: meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Meta AI Blog: Introducing Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- [HF Blog: Welcome Llama 3](https://huggingface.co/blog/llama3)
- [Detecting Stylistic Fingerprints (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)
- [Linguistic Characteristics Survey (arXiv:2510.05136)](https://arxiv.org/abs/2510.05136)
- [Why Does ChatGPT "Delve" So Much? COLING 2025](https://aclanthology.org/2025.coling-main.426.pdf)
- [Llama 3.1 Model Card & Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)

---

## 4. `llama-3.3-70b-instruct`

### Architecture & Training

- Same decoder-only family as Llama 3.1, 70B params, GQA, 128k context. Released Dec 2024. Knowledge cutoff Dec 2023. Training: 7.0M GPU-hours on H100-80GB.
- **Tokenizer:** identical tiktoken 128,256-token vocabulary as Llama 3.1 8B. Minor difference: 8B has `model.ignore_merges: true`, 70B does not — subtle tokenization difference for edge-case inputs.
- **Post-training:** same pipeline (SFT → rejection sampling → DPO), but 25M+ synthetic examples generated by 405B model for 70B variant — stronger teacher than for 8B.
- Primary improvement over Llama 3.1 70B: more DPO iterations, higher-quality preference data for math/coding/reasoning. Stylistic differences from Llama 3.1 70B are modest — same family, better calibration.

### Stylistic Tendencies

- **All Llama 3.x patterns apply** — executed with greater coherence, length, and elaboration than 8B.
- **More verbose than 3.1 8B:** more context, caveats, and elaborations → longer responses, more hedging qualifiers (size-driven).
- **High IFEval score (92.1):** best-in-class instruction following → style-mimicry prompts most effective at evasion on this model.
- **Structural tendencies:**
  - intro sentence → body (bulleted/numbered) → closing synthesis starting with "In summary," or "Overall,"
  - Headers (`## Section Name`) used aggressively for longer responses
  - Code always in fenced blocks, even single-line
- **Clichés:**
  - "Delve into" (persists)
  - "It's worth noting," "It's important to emphasize" (persist)
  - "A comprehensive overview," "a nuanced understanding," "a robust approach"
  - "In today's rapidly evolving landscape"
  - Closings: "I hope this helps!", "Let me know if you'd like to explore any aspect further"
  - Transitions: "Moving on to…", "Building on that…", "To put this in perspective…"
- **Safety refusals:** more nuanced than Llama 2 — partial answer + inline flag: "While I can help with [X], I'd note that [concern]…" Detectable pattern distinct from both Llama 2 (block refusal) and GPT-4 (more variable).

### Detection-Relevant Features

- **Per-token entropy:** 70B DPO is highly calibrated — bimodal per-token surprisal distribution. PAWN framework (ScienceDirect 2025) specifically targets this profile.
- **Burstiness:** slightly higher than 8B (capacity for long complex sentences), but markdown suppresses it within list-heavy responses. Detectable burstiness discontinuities at paragraph ↔ list boundaries.
- **Text length:** measurably longer average responses than GPT-4o-mini for equivalent prompts (FAID paper).
- **N-gram fingerprints:**
  - "it's worth" (very high freq bigram prefix)
  - "in summary" / "to summarize" (closing anchors)
  - "key takeaways" (list header cliché)
  - "for example," + comma-separated list (structural tic)
  - "overall," as paragraph opener
- **Absent features (useful negative signals):** typos, self-corrections, informal abbreviations, sentence fragments for rhetorical effect, regional vocabulary/slang, personal anecdotes, topic digressions.

### Detection Results & Vulnerabilities

- arXiv:2503.01659: Llama family detectability confirmed at high precision. Model-level attribution (70B vs 8B within Llama 3) is harder than family-level.
- FAID (arXiv:2505.14271): **88.5% accuracy** in LLM-family attribution including Llama-3.3-70B-Instruct-Turbo. Text length distribution, n-gram distribution, and semantic embedding all contributed — no single feature sufficient.
- arXiv:2509.18880 (Diversity Boosts Detection): diverse detector ensembles improve detection of strong models like Llama 3.3 70B that evade simple single-metric detectors.
- **Vulnerability:** strong DPO instruction following → responds well to style-mimicry system prompts ("write as a tired student"). Underlying perplexity/structural signals harder to mask.

### References

- [Llama 3 Herd paper (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [HF: meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Llama 3.3 Model Card & Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/)
- [FAID: Fine-Grained AI Detection (arXiv:2505.14271)](https://arxiv.org/abs/2505.14271)
- [Detecting Stylistic Fingerprints (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)
- [Diversity Boosts AI Detection (arXiv:2509.18880)](https://arxiv.org/abs/2509.18880)
- [PAWN: Not All Tokens Are Equal (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S156625352500538X)
- [GPTZero AI Vocabulary](https://gptzero.me/ai-vocabulary)

---

## Cross-Model Comparison

| Feature | llama-2-7b-chat | llama-2-70b-chat | llama-3.1-8b-instruct | llama-3.3-70b-instruct |
|---|---|---|---|---|
| Tokenizer | SentencePiece 32k | SentencePiece 32k | tiktoken 128k | tiktoken 128k |
| Context window | 4,096 | 4,096 | 128,000 | 128,000 |
| Alignment method | SFT + PPO | SFT + PPO | SFT + rejection + DPO | SFT + rejection + DPO |
| Sycophancy level | High | Very high | Low–medium | Low |
| Safety phrase overuse | High | Very high | Low | Low |
| Downtoner use | High | High | Low (documented) | Low (documented) |
| Markdown/list use | Moderate | Moderate | High | Very high |
| Sentence burstiness | Low | Low | Low–moderate | Moderate |
| "Certainly/Absolutely" openers | Very high | Very high | Moderate | Low |
| "Delve/nuanced/robust" | Moderate | Moderate | High | High |
| "In summary/Overall" closings | High | High | High | Very high |
| Evasion via style prompt | Low | Low–medium | Moderate | High (IFEval 92.1) |

---

## Detection Strategy by Sub-Family

**Llama 2 Chat (7B and 70B):**
- Safety/hedge phrase frequency is the strongest feature: "it's worth noting," "as an AI," "I'd be happy to," "Of course!"
- Low burstiness (sentence-length std dev) is a reliable secondary feature
- Downtoner frequency ("somewhat," "slightly," "rather")
- SentencePiece tokenization artifacts (white-box settings)
- `[INST]`/`<<SYS>>` template artifacts are hard identifiers if present

**Llama 3.x Instruct (8B and 70B):**
- Unprompted markdown structure — bullet lists, bold headers, fenced code blocks
- High-frequency markers: "Delve," "nuanced," "multifaceted," "comprehensive," "robust," "pivotal"
- "In summary" / "Overall" / "Key takeaways" as closing anchors
- Absence of downtoners (contrasts with Llama 2)
- tiktoken 128k vocabulary produces different byte-pair boundaries (white-box perplexity scoring)
- Per-token entropy profile (PAWN-style) effective at 70B due to DPO calibration

**Cross-family shared signals:**
- Fewer adjectives than human text
- Formal, Latinate vocabulary preferences
- Higher lexical diversity than Falcon/Mistral but lower than human-written text
- No typos, self-corrections, or rhetorical sentence fragments
