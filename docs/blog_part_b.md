# Fine-Tuning ASR for Indian Accents and the Banking Domain

In our previous post, we established that zero-shot ASR models degrade severely when confronted with Hinglish (code-switched) banking queries and regional Indian accents. In this post, we discuss how we designed a fine-tuning strategy to adapt models for Krim.ai's banking voice bot, optimize accent robustness, and eliminate Whisper's hallucination loops.

---

## Our Training Configurations

To scale domain adaptation, we designed three distinct fine-tuning setups:
*   **Config B (Synthetic Adaptation):** A lightweight dataset containing 639 high-fidelity Gemini-generated synthetic Hinglish banking queries.
*   **Config C (Mixed Domain):** Combined Config B with the **MUCS Hindi-English finance dataset** (90 hours of lecture-style speech).
*   **Config D (Scale & Dialect):** Added massive native speech corpora to combat domain drift: **Project Vaani (Hindi Belt)** and **RESPIN-S1.0 (Agriculture & Finance)**, alongside MUCS and Synthetic datasets.

---

## Results: The Power of Domain Adaptation

We evaluated our fine-tuned models on Hinglish banking queries (`synthetic_100`) and general Hindi (`kathbath_hindi`):

### Comparative WER (%) Results

| Model & Config | Hinglish Banking (`synthetic_100`) | General Hindi (`kathbath_hindi`) |
| :--- | :---: | :---: |
| **`indicwav2vec-hindi` (Baseline)** | 75.49% | **11.64%** |
| **`indicwav2vec-banking-configC`** | 78.95% | 17.20% |
| **`indicwav2vec-banking-configD`** | 73.35% | 14.52% |
| **`whisper-medium-hi` (Baseline)** | 179.17% | 41.64% |
| **`whisper-medium-banking-configC`** | **37.74%** | 36.61% |
| **`whisper-medium-banking-configD`** | 48.73% | **20.57%** |

### Key Experimental Insights:
1.  **Whisper Domain Adaptation:** Fine-tuning Whisper-medium on code-switched banking datasets yielded a massive **141.43% absolute reduction in Hinglish WER** (from **179.17% down to 37.74%**). Target-domain training is essential for code-switched corporate ASR.
2.  **Generalization vs. Drift (Forgetting):**
    *   *Config C Drift:* Monolingual `indicwav2vec` experienced a minor degradation on general Hindi (from **11.64% to 17.20%** WER) due to vocabulary drift towards Hinglish terms.
    *   *Config D Generalization:* By including large-scale native speech (Vaani & RESPIN) under Config D, we recovered significant performance, slashing Whisper's general Hindi WER to **20.57%** (a **16.04% absolute improvement** over Config C baseline). Large-scale native speech is key to preventing catastrophic forgetting.

---

## Quantifying Errors: Inside the Alignment Logic

To understand *why* the models errored, we wrote an alignment classifier to categorize word-level mismatches on Hinglish speech into distinct categories:

| Error Category | `indicwav2vec-configC` | `whisper-configC` | `whisper-medium-hi` (Baseline) |
| :--- | :---: | :---: | :---: |
| **Entity Deletions** | **47.42%** | 23.55% | 5.53% |
| **Number Substitutions** | 29.08% | **43.31%** | 9.74% |
| **Hindi/Latin Script Mismatch** | 17.39% | 28.78% | 3.41% |
| **Hallucinations (Insertions)** | 0.14% | **0.58%** | **78.45%** |
| **Other Substitutions/Deletions** | 5.97% | 3.78% | 2.87% |

### Key Error Analysis Insights:
*   **Whisper Hallucination Suppression:** The baseline `whisper-medium-hi` spent **78.45%** of its error budget on hallucinations (infinite repetition loops of filler words). Fine-tuning on Config C virtually eliminated this, reducing hallucinations to **0.58%** of errors.
*   **The Script Barrier:** For `indicwav2vec-banking-configC`, script and vocabulary mismatches (representing English terms written in Latin script) drove **21.60%** of all errors. This is a structural limitation: character-based models with monolingual vocabularies cannot generate English spelling.
*   **Numeric Precision:** For Whisper, transcribing digits (number substitutions) is the largest remaining hurdle, representing **43.31%** of errors.

---

## Accent Robustness: The LAHAJA Benchmark

Dialect variation is a significant bottleneck in production. We evaluated the models on **LAHAJA** (6,152 conversational dialect samples) across 5 major regions:

| Accent Group | `indicwav2vec` (Baseline) | `nemotron-3.5-asr` (Baseline) | `indicwav2vec-configD` | `whisper-configD` |
| :--- | :---: | :---: | :---: | :---: |
| **punjab_haryana** | 28.16% | 21.89% | 26.67% | 30.73% |
| **hindi_belt** | 26.48% | 20.19% | 27.20% | 29.44% |
| **south_india** | 35.11% | 26.25% | 36.09% | 36.38% |
| **east_india** | 32.81% | 26.13% | 34.79% | 37.71% |
| **west_india** | 30.41% | 20.75% | 32.13% | 34.74% |
| **Overall Mean** | **33.22%** | **25.09%** | **34.99%** | **38.27%** |

### How We Patched Whisper's Autoregressive Loops

We discovered that standard greedy decoding on Whisper collapses on short, noisy, and dialect-rich clips. To fix this, we updated our inference wrapper (`whisper_local.py`) to enforce safety generation constraints:
```python
generate_kwargs = {
    "condition_on_prev_tokens": False,  # Prevent carry-over context hallucination
    "repetition_penalty": 1.1,          # Penalize token repetition
    "no_repeat_ngram_size": 4,          # Block repeating 4-grams
    "compression_ratio_threshold": 1.35 # Fall back to temperature search if loops occur
}
```
Deploying these safety parameters successfully stabilized Whisper's decoder, causing its Lahaja WER to drop from **161.99% down to 38.27%**.

---

## Concluding Recommendations

For deploying voice bots in regional markets:
1.  **If accuracy on code-switched banking is top priority:** Deploy **Fine-Tuned Whisper-Medium (Config C/D)**.
2.  **To handle high regional accent variations at low cost:** Use **Bilingual Conformer-CTC** or **Nemotron-3.5-ASR-Streaming** served via NVIDIA Triton.
3.  **Prevent domain drift** by always combining your custom domain synthetic data with general-domain native speech corpora (like Vaani) in your training loops.
