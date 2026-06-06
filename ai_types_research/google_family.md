# Google AI Models — Detection-Oriented Research

## Overview

Covers five Google/Gemini models in the PAN25 dataset. Gemini Pro (1.0) and 1.5 Pro share similar stylistic fingerprints — polished neutrality, encyclopedic tone, low burstiness. Gemini 2.0 Flash trades depth for conciseness. text-bison-002 is PaLM 2-based and now deprecated. The paraphrase variant is the hardest detection case and is analyzed in depth.

---

## 1. `gemini-pro` (Gemini 1.0 Pro, December 2023)

### Architecture & Training

- Decoder-only natively multimodal Transformer (text + image + audio + video joint pretraining). Released Dec 2023, replaced PaLM 2 in Bard (renamed Gemini in Feb 2024).
- Post-training: SFT on human-curated instruction pairs → RLHF (PPO) with reward models covering helpfulness, factuality, and safety.
- Context: ~32K tokens. Parameter count undisclosed.

### Stylistic Tendencies

- **Conversational register over academic register** — contrasts with ChatGPT. Research (Rudnicka, Univ. Gdańsk, via Scientific American 2025) using the Delta method found Gemini's trigrams are explanatory and conversational: "the way for," "the cascade of," "high blood sugar," while ChatGPT leans clinical/formal ("blood glucose levels," "characterized by elevated"). Gemini chose "sugar" 158x vs ChatGPT's 25x in matched medical text.
- **Hedging language as dominant surface marker:**
  - "It is worth noting that..." (~4x more than human writers)
  - "In the realm of...", "In the context of..."
  - "It's important to consider", "This underscores the importance of..."
  - "Plays a crucial role", "A multifaceted approach"
- **Polished Wikipedia-like neutrality:** safe, generic, politically neutral — Originality.AI describes it as resembling Wikipedia prose.
- **Predictable structure:** evenly-sized sections, consistent intro/conclusion, frequent lists as default organizational device.
- **Absence of informal markers:** no em-dashes used casually, no sentence fragments, very rare informal contractions.

### Detection-Relevant Features

- **Perplexity:** very low — defaults to statistically safest next token at every step. Commercial detectors flag raw Gemini Pro text at 84–98% accuracy.
- **Burstiness:** very low (0.15–0.22 range vs human baseline >0.30). Sentence lengths and structural complexity are highly uniform.
- **TTR:** reduced vs human writing — same concept-adjacent vocabulary recurs across paragraphs.
- **Cliché openers:** "In today's fast-paced world" ~107x more frequent in AI text. "Furthermore," "Moreover," "Additionally," "In conclusion" all statistically overrepresented.
- **POS bigram patterns:** average sentence length and POS bigram frequencies are the most powerful discriminative features (ResearchGate 2024 stylometric detection study).
- **Punctuation:** perfect, consistent Oxford commas, no stray whitespace, no em-dash vs hyphen confusion.

### Detection Results & Vulnerabilities

- Originality.AI (2024): detection accuracy on Gemini Pro aligned with GPT-3.5/GPT-4 tier — not significantly harder or easier.
- Turnitin claims 98% accuracy on raw Gemini output.
- arXiv:2503.01659: ensemble of three classifiers achieves precision 0.9988, FPR 0.0004 across Claude/Gemini/Llama/OpenAI families. Gemini's fingerprint is persistent and consistent.
- **Soft vulnerability:** explicit style forcing ("write informally," "use contractions") reduces detector confidence but does not reach human-text levels.

### References

- [Can Google Bard Gemini Pro Content Be Detected? — Originality.AI](https://originality.ai/blog/google-bard-gemini-pro-ai-detection)
- [ChatGPT and Gemini AIs Have Uniquely Different Writing Styles — Scientific American](https://www.scientificamerican.com/article/chatgpt-and-gemini-ai-have-uniquely-different-writing-styles/)
- [Detecting Stylistic Fingerprints of LLMs (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)
- [PAN24 Voight-Kampff Dataset — Zenodo](https://zenodo.org/records/10718757)

---

## 2. `gemini-1.5-pro`

### Architecture & Training

- Released Feb 2024. **Mixture-of-Experts (MoE)** architecture — major departure from 1.0 Pro's dense transformer.
- Context: up to 2M tokens (1M at launch). Near-perfect needle-in-haystack retrieval (>99.7% at 1M tokens).
- Multi-stage training: pretraining (multimodal) → SFT → RLHF + reward models → safety filtering.
- Parameter count undisclosed; MoE means total params high but per-token compute much lower.

### Stylistic Tendencies

- **Longer, more elaborated responses by default** than 1.0 Pro — more detail, more multi-clause sentences, but same underlying hedging patterns.
- **More concise and accurate summaries** than the Flash variant.
- **Encyclopedic tone reinforced** — higher information density than 1.0 Pro, writes more like a long-form Wikipedia article.
- **Structural uniformity persists:** even with 2M token context, paragraph unit length and complexity variance within an output remain uniform (low burstiness). Detectable: style does not drift across a long 30-paragraph output, unlike human writers.
- **Same hedging phrases** as 1.0 Pro (these come from RLHF reward modeling, not architecture).

### Detection-Relevant Features

- **Perplexity:** low, similar to 1.0 Pro. Originality.AI found detection rates of 84–98% persist for unmodified 1.5 Pro outputs.
- **Burstiness:** 0.15–0.22 range — unchanged despite MoE architecture.
- **Absence of stylistic drift:** uniform prose quality, hedging rate, and sentence length from paragraph 1 to paragraph 30 in a long output. Strong distributional signal.
- **Same cliché set** as 1.0 Pro: "It is worth noting," "In the realm of," "This underscores," "Plays a crucial role," "A multifaceted approach."

### Detection Results & Vulnerabilities

- Originality.AI: 1.5 Pro content remains detectable at similar rates to 1.0 Pro — the 2M-token context window did not change the stylistic fingerprint.
- arXiv:2503.01659: Gemini-family texts across multiple generations classified at near-perfect precision (0.9988).
- GPTZero and Turnitin report similar rates on 1.5 Pro as on 1.0 Pro.
- **Vulnerability:** improved instruction-following means explicit style forcing ("write informally," "use contractions," "vary sentence length") is more effective at evasion than on 1.0 Pro — still not human-text level.

### References

- [Is Google Gemini Pro 1.5 Content Detectable? — Originality.AI](https://originality.ai/blog/google-gemini-pro-content-detectable)
- [Gemini 1.5 Technical Report (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Detecting Stylistic Fingerprints of LLMs (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)

---

## 3. `gemini-2.0-flash`

### Architecture & Training

- Released Dec 2024 (experimental), stable Feb 2025. Dense transformer (~18B active params), not MoE.
- Architecture: 42-layer Transformer, hidden dim 2,048, 32 attention heads, FlashAttention with grouped QKV, RoPE. Context: 1M tokens.
- Built for the "agentic era" — native tool use, multimodal generation, low latency (55ms/token vs 110ms for 2.5 Flash).
- Post-training: RL against verifiable signals for the Thinking variant; standard RLHF/SFT for base 2.0 Flash.
- Optimized for speed and cost at lower quality tier than 2.0 Pro.

### Stylistic Tendencies

- **More conversational and engaging than Pro variants** — "wittier," more informal phrasing. In creative writing, Flash explores internal conflict vs Pro's structured narrative.
- **Concise by default** — shorter output lengths than Pro variants; less padding. This reduces occurrence of characteristic hedging phrases.
- **Slight reduction in hedging phrase density** — "It is worth noting" and "In the realm of" are present but less frequent (shorter default outputs).
- **Technical responses less comprehensive** — slightly less detail than Pro on the same prompt. Paradoxically makes it slightly harder to distinguish from human writing on short samples.
- **Retained Gemini family vocabulary:** "sugar" over "glucose," conversational framing, absence of informal register markers.

### Detection-Relevant Features

- **Shorter outputs introduce detection noise on short samples** — stylometric methods need longer passages. A 200-word Flash output is harder to classify than a 600-word Pro output.
- **Low burstiness persists** — structural uniformity does not disappear even in shorter outputs.
- **Consistent punctuation patterns** — no stray punctuation, consistent Oxford comma, no typos.
- **Speed-quality tradeoff:** faster generation at lower temperature → even higher local token predictability than Pro-class models.
- **Tool-use artifacts:** in agentic contexts, outputs often include structured action-result formatting and JSON-schema responses — trivially detectable as AI in those contexts.

### Detection Results & Vulnerabilities

- No specific published detection accuracy study isolated to Gemini 2.0 Flash as of April 2026.
- Conciseness creates mild evasion advantage on **short text tasks** — burstiness/perplexity approaches require minimum ~150–200 words to be reliable.
- arXiv:2506.07001 (Adversarial Paraphrasing, NeurIPS 2025): paraphrasing under guidance of a detector reduces detection by avg 87.88% across all detectors — applicable to all Gemini variants.

### References

- [Gemini 2.0 Flash Technical Docs — Google Cloud](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash)
- [Gemini 2.0 Family Expands — Google Developers Blog](https://developers.googleblog.com/en/gemini-2-family-expands/)
- [Adversarial Paraphrasing (arXiv:2506.07001)](https://arxiv.org/abs/2506.07001)

---

## 4. `text-bison-002` (PaLM 2, Google Cloud Vertex AI)

### Architecture & Training

- Fine-tuned variant of **PaLM 2**. PaLM 2 Technical Report (arXiv:2305.10403) specs: decoder-only Transformer, **SentencePiece 256K-token vocabulary** (much larger than GPT's ~50K — better multilingual/code coverage). RoPE positional embeddings. No biases in dense kernels or layer norms. Pretrained on 780B tokens of high-quality multilingual text.
- `text-bison` fine-tuned specifically for **single-turn text completion** — not dialogue. Classification, summarization, extraction, content creation.
- `@002` is the second stable version with "improved prompt responses." Max input: 8,192 tokens.
- **Now deprecated:** text-bison@001 retired June 2024.

### Stylistic Tendencies

- **Over-formal register as default:** example output from comparative eval: "Today, as we stand on the precipice of a new era, we find ourselves at a crossroads of immense technological advancements" — purple, rhetorically inflated.
- **Informative over conversational:** better at generating "informative text," with encyclopedic/didactic sentence structure rather than narrative or exploratory prose.
- **Self-contained, non-conversational:** optimized for single-turn completion — outputs are complete closed units that do not trail off or invite follow-up.
- **List tendency:** defaults to bulleted or numbered lists for open-ended information questions — noted consistently in third-party usage reports.
- **Conservative token selection despite large vocabulary:** 256K vocab access to rare/technical words, but fine-tuning steers toward predictable high-frequency vocabulary — same "lexically predictable" detection signal as other aligned models.

### Detection-Relevant Features

- **Perplexity:** low — same fundamental issue as all RLHF models. Lower perplexity, more uniform sentence structures, higher lexical repetitiveness vs human text.
- **Burstiness:** very low — fine-tuned for task-completion rather than conversational flow → every sentence roughly same length and information density.
- **256K tokenizer artifact:** large SentencePiece vocabulary means less aggressive tokenization than GPT-based models. Compound words and rare morphological forms tokenized as single tokens — subtle distributional difference from OpenAI-family outputs.
- **3-part macro-structure:** "In the realm of [topic]…" → structured enumeration → summary sentence. Consistent across prompts.
- **No casual register:** contractions rare or absent, no sentence fragments, consistently neutral-to-formal tone.

### Detection Results & Vulnerabilities

- text-bison-002 is deprecated and rarely appears in 2024–2026 detection benchmark datasets.
- **256K vs ~50K vocab** creates measurable character-level entropy and subword boundary differences — exploitable by forensic stylometry but not by most commercial detectors.

### References

- [PaLM 2 Technical Report (arXiv:2305.10403)](https://arxiv.org/pdf/2305.10403)
- [PaLM 2 Models Overview — Google AI](https://ai.google.dev/palm_docs/palm)
- [Epic Battle of AI Models — Bito.ai](https://bito.ai/blog/epic-battle-of-ai-models-google-bard-chatgpt-3-5-gpt-4-bison-palm-2-and-anthropic-claude-unveiling-the-best/)

---

## 5. `gemini-pro-paraphrase` (Gemini Pro text rewritten to sound human)

### What This Is

Not a distinct model. A **dataset condition** in the PAN 2024/2025 Voight-Kampff task (Zenodo:10718757): Gemini Pro-generated articles subsequently paraphrased — by another LLM or a dedicated paraphraser — to obscure their origin. Represents the hardest adversarial detection case.

### What Paraphrasing Removes

1. **Surface lexical markers:** clichés ("It is worth noting that," "In the realm of") are the first casualties. One pass through a paraphrase model replaces most regex-matchable clichés.
2. **N-gram based detection signals:** commercial detectors relying on high-frequency AI n-gram patterns are significantly degraded. Krishna et al. NeurIPS 2023 (arXiv:2303.13408): DIPPER paraphrasing drops DetectGPT accuracy from 70.3% to 4.6% at 1% FPR.
3. **Vocabulary predictability at word level:** synonym substitution increases local perplexity, pushing output toward human-writing's high-perplexity zone.
4. **Watermark signals:** all current soft watermarking approaches are defeated.

### What Paraphrasing Does NOT Remove

1. **Deep syntactic structure:** clause ordering, subordination patterns, passive-voice frequency, and sentence segmentation survive synonym swaps.
2. **Burstiness profile:** sentence-length distribution and structural uniformity (0.15–0.22) persist through paraphrasing unless the paraphraser was explicitly instructed to vary lengths.
3. **Discourse structure / macro organization:** 3-part macro-structure (introduction hedge → enumerated body → summary conclusion) survives — it is a property of information organization, not lexical choice.
4. **Semantic coherence patterns:** AI's characteristic topic progression and co-reference patterns differ from human writing and survive paraphrasing.
5. **AI-paraphrased fingerprint:** when an AI paraphraser is used on AI source text, the output carries the paraphraser's stylistic fingerprint — "AI-humanized text is an AI-humanized fingerprint easily distinguishable from authentic human variation."

### Effect on Different Detector Types

| Detector type | Effect of paraphrase |
|---|---|
| Regex/n-gram phrase matching | Strongly degraded — clichés replaced |
| Perplexity-based (DetectGPT) | Severely degraded — accuracy drops 60–90% |
| Watermark-based | Completely defeated |
| Supervised classifier (fine-tuned RoBERTa/DeBERTa) | Partially degraded — 20–40% accuracy loss |
| Stylometric ensemble (burstiness + syntax + discourse) | Least degraded — retains sentence structure + discourse signals |
| Retrieval-based defense | Highly effective — 80–97% detection at 1% FPR (Krishna et al.) |

### Persistent Detection Signals After Paraphrasing

1. Low burstiness (sentence length variance below human baseline)
2. Absence of first-person hedging, emotional language, personal anecdote
3. Consistent punctuation hygiene — no typos, no colloquial punctuation
4. Linear, predictable semantic coherence without surprise (no tangents, contradictions, irony)
5. Lack of stylistic register variation across paragraphs
6. POS bigram distributions — grammatical structure preserved through paraphrasing

### Practical Implications

- Do not rely on phrase/vocabulary features against paraphrased text — they will fail.
- **Prioritize:** burstiness, sentence-length variance, syntactic feature distributions, discourse-level structure.
- Train on paraphrased AI text explicitly — the PAN dataset provides `gemini-pro-paraphrase` as a condition precisely for this.
- Use retrieval as a defense layer when you have access to candidate model outputs.

### References

- [Paraphrasing Evades Detectors, Retrieval Is Effective Defense (arXiv:2303.13408)](https://arxiv.org/abs/2303.13408)
- [Adversarial Paraphrasing: Universal Attack (arXiv:2506.07001)](https://arxiv.org/abs/2506.07001)
- [PAN24 Voight-Kampff Dataset — Zenodo](https://zenodo.org/records/10718757)
- [PAN24 Task Overview — CEUR-WS Vol-3740](https://ceur-ws.org/Vol-3740/paper-225.pdf)
- [Why Perplexity and Burstiness Fail to Detect AI — Pangram Labs](https://www.pangram.com/blog/why-perplexity-and-burstiness-fail-to-detect-ai)

---

## Cross-Model Comparison

| Model | Perplexity | Burstiness | Top Clichés | Raw Detection Accuracy | Hardest For |
|---|---|---|---|---|---|
| gemini-pro (1.0) | Very low | 0.15–0.22 | "It is worth noting," "In the realm of," "This underscores" | 84–98% | Short texts (<150 words) |
| gemini-1.5-pro | Very low | 0.15–0.22 | Same + encyclopedic elaboration | 84–98% | Explicit informal prompting |
| gemini-2.0-flash | Low | 0.15–0.22 | Same family, lower density | Not published | Short outputs |
| text-bison-002 | Low | Low | Formal openers, list structure | High | Technical domain text |
| gemini-pro-paraphrase | Elevated (after paraphrase) | 0.15–0.22 (persists) | Clichés removed; structure survives | 10–60% | Phrase/n-gram detectors; perplexity-only |

---

## Sources

- [Originality.AI Gemini Pro Detection](https://originality.ai/blog/google-bard-gemini-pro-ai-detection)
- [Originality.AI Gemini 1.5 Detection](https://originality.ai/blog/google-gemini-pro-content-detectable)
- [Scientific American: ChatGPT vs Gemini Writing Styles](https://www.scientificamerican.com/article/chatgpt-and-gemini-ai-have-uniquely-different-writing-styles/)
- [Detecting Stylistic Fingerprints of LLMs (arXiv:2503.01659)](https://arxiv.org/abs/2503.01659)
- [Paraphrasing Evades Detectors (arXiv:2303.13408)](https://arxiv.org/abs/2303.13408)
- [Adversarial Paraphrasing (arXiv:2506.07001)](https://arxiv.org/abs/2506.07001)
- [PAN24 Voight-Kampff Dataset (Zenodo:10718757)](https://zenodo.org/records/10718757)
- [Feature-Based Detection of AI Text — ResearchGate](https://www.researchgate.net/publication/398588043_Feature-Based_Detection_of_AI-Generated_Text_An_Analysis_of_Stylometric_and_Perplexity_Markers_in_Contemporary_Large_Language_Models)
- [PaLM 2 Technical Report (arXiv:2305.10403)](https://arxiv.org/pdf/2305.10403)
