# WER Is Not Enough: Benchmarking ASR for Indian Banking

In the world of voice bots and automated customer care, Automatic Speech Recognition (ASR) is the front door. If the ASR model cannot understand the customer, the entire downstream system—conversational AI, slot filling, sentiment analysis—falls apart.

When scaling a voice bot to handle **500+ concurrent calls**, traditional system benchmarks often fall back on a single number: **Word Error Rate (WER)**. However, in bilingual and dialect-rich regions like India, WER alone is a dangerously incomplete metric.

In this post, we share our findings from benchmarking several state-of-the-art ASR models (including Whisper, Conformer-CTC, and Nemotron) on Indian English/Hindi and code-switched banking queries.

---

## The Multilingual Challenge in Indian Banking

Indian conversational speech is rarely monolingual. A customer calling a banking bot is highly likely to speak in **Hinglish** (a blend of Hindi and English). They will say sentences like:
* *"Mera credit card block ho gaya hai, please help me activate it."*
* *"EMI payment date kya hai?"*
* *"Account statement send kar dijiye."*

This presents three unique challenges for ASR models:
1. **The Script Mismatch:** Standard Hindi models are trained on Devanagari script (e.g., *"क्रेडिट कार्ड"*), while English models expect Latin script (e.g., *"credit card"*). Standard character-based monolingual models have vocabulary constraints that prevent them from outputting Latin characters, forcing spelling substitution errors.
2. **Numeric Precision:** In banking, missing a single digit in a phone number, OTP, account ID, or amount can be catastrophic. Standard WER treats a substituted digit the same as a spelling mistake, hiding critical failures.
3. **Domain Vocabulary:** Named entities like "CIBIL", "Aadhaar", "KYC", or bank names ("HDFC", "SBI") must be transcribed with 100% accuracy to trigger the correct intent.

---

## Beyond WER: A Multi-Dimensional Metric Suite

To evaluate models properly, we constructed a multi-dimensional metric framework:
* **Word Error Rate (WER) & Character Error Rate (CER):** For general text transcription quality.
* **Number Error Rate (NER):** Measures the precision of transcribing numbers and digits.
* **Entity Accuracy:** Measures the percentage of core banking entities (e.g., *"OTP"*, *"KYC"*, *"PIN"*) transcribed correctly.
* **Real-Time Factor (RTF):** The ratio of processing latency to audio duration. An RTF < 1.0 is required for real-time applications; lower is better.

---

## Zero-Shot Benchmark Results

Here is how the baseline (zero-shot) models compared on general Hindi (**Kathbath**) and code-switched banking speech (**Synthetic 100**):

### General Hindi (Kathbath) Performance

| Model | Architecture | WER (%) | CER (%) | RTF |
| :--- | :--- | :---: | :---: | :---: |
| **`indicwav2vec-hindi`** | Wav2Vec 2.0 (Mono) | **11.64%** | **3.30%** | **0.013** |
| **`nemotron-3.5-asr`** | Transducer (RNNT) | 13.00% | 4.44% | 0.174 |
| **`stt-hi-conformer-ctc-large`** | Conformer CTC | 13.26% | 3.99% | **0.013** |
| **`whisper-medium-hi`** | Encoder-Decoder | 41.64% | 15.85% | 0.264 |
| **`whisper-large-v3-turbo`** | Encoder-Decoder | 32.01% | 11.20% | 0.084 |

### Hinglish Banking (Synthetic 100) Performance

| Model | WER (%) | NER (%) | Entity Accuracy (%) | RTF |
| :--- | :---: | :---: | :---: | :---: |
| **`indicwav2vec-hindi`** | 75.49% | 90.91% | 9.09% | 0.114 |
| **`whisper-medium-hi`** | 179.17% | 92.93% | 7.07% | 0.275 |
| **`stt-hi-conformer-ctc-large`** | 73.61% | 87.25% | 12.75% | 0.013 |
| **`nemotron-3.5-asr`** | 67.81% | — | — | 0.174 |

### Critical Takeaways:
1. **The Script Barrier:** On pure Hindi (Kathbath), `indicwav2vec-hindi` is exceptional (11.64% WER). But on Hinglish banking queries, its WER skyrockets to **75.49%**. It cannot output English letters, so it spells "credit card" as phonetically distorted Devanagari or drops it entirely.
2. **Whisper Hallucination Loop:** Baseline `whisper-medium-hi` collapses on conversational Hinglish, scoring **179.17% WER**. Due to background noise, brief pauses, and non-standard syntax, Whisper's autoregressive decoder falls into infinite repetition loops or deletes entire phrases.

---

## Quality vs. Latency: The Pareto Frontier

For a voice bot running at scale, compute budget and latency are as critical as accuracy. We plotted the **Quality-Latency Pareto Frontier** (WER vs. RTF) to find the optimal trade-offs:

![Pareto Frontier](results/plots/kathbath_pareto.png)

From our Pareto sweep, three champions emerge:
* **The Latency Champion:** `stt-hi-conformer-ctc-large`. With a mean chunk latency of **49ms (RTF 0.013)**, this non-autoregressive CTC model runs almost instantaneously. It uses minimal VRAM and is highly scalable.
* **The Accuracy-Speed Balance:** `whisper-large-v3-turbo`. At an RTF of **0.084**, it offers 4x faster execution than `whisper-medium` while maintaining competitive zero-shot WER.
* **The Streaming Champion:** `nemotron-3.5-asr-streaming-0.6b`. Operating at **0.174 RTF**, this model processes audio chunk-by-chunk in real time as the user speaks. The final perceived latency for the user is virtually zero since transcription happens concurrently with speech.

---

## Designing for Scale (500+ Concurrent Calls)

If you are designing the ASR loop for a customer voice bot, here is our architectural blueprint:

1. **Avoid heavy autoregressive decoders (like Whisper-large or Voxtral) in live loops.** While accurate offline, they are computationally expensive and hard to scale to hundreds of parallel streams.
2. **For live interactive bots, deploy Nemotron-3.5-ASR-Streaming-0.6b.** Its native streaming architecture ensures real-time slot filling and dynamic turn-taking.
3. **For maximum concurrency and cost efficiency, deploy Conformer-CTC Large.** served on **NVIDIA Triton Inference Server** with TensorRT-ASR. This non-autoregressive setup parallelizes hundreds of incoming audio chunks on a single GPU, keeping server costs extremely low.

In our next post, we will show how we broke through the Hinglish script barrier and solved Whisper's repetition loops using domain-specific fine-tuning. Stay tuned!
