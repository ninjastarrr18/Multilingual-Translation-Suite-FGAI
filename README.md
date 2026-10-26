# Multilingual-Translation-Suite-FGAI

## Problem Statement and Objective
The objective of this project is to develop, evaluate, and compare machine translation systems across encoder-decoder and decoder-only architectures. The system translates English text to French using four foundational generative AI models.

## Dataset Description
- **Source:** Hugging Face `opus_books` (English-French).
- **Link:** [OPUS Books on Hugging Face](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
- **Description:** A parallel corpus created from copyright-free books. A subset of sentence pairs was utilized and split into training and test sets to manage computational constraints.

## Models Implemented
1. **Basic Transformer (Helsinki-NLP/opus-mt-en-fr):** A standard encoder-decoder architecture fine-tuned for translation.
2. **mT5 (google/mt5-small):** A multilingual text-to-text transformer fine-tuned using a prefix-based approach.
3. **LLaMA (TinyLlama/TinyLlama-1.1B-Chat-v1.0):** An ungated, smaller 1.1B parameter variant of the LLaMA decoder-only architecture utilized to accommodate computational limits.
4. **Gemma (Qwen/Qwen2.5-1.5B-Instruct):** A lightweight decoder-only variant used as an ungated alternative for prompted generation.

## Execution Steps and Dependencies
**Dependencies:** `torch`, `transformers`, `datasets`, `evaluate`, `sacrebleu`, `bitsandbytes`, `accelerate`.
**Execution:**
1. Install requirements via pip.
2. Run the Jupyter Notebook cells sequentially to download datasets, train encoder-decoder models, and generate prompted translations from decoder models.

## Results Obtained & Model Comparison
| Model | BLEU | chrF |
|---|---|---|
| Basic Transformer | 21.49 | 47.07 |
| mT5 | 0.00 | 0.00 |
| LLaMA (TinyLlama) | 2.52 | 28.47 |
| Gemma (Qwen) | 2.46 | 26.20 |

**Comparison:** The Basic Encoder-Decoder Transformer proved the most stable and efficient for translation. The mT5 model demonstrated fp16 mixed-precision instability (gradient overflow resulting in NaN loss and a 0.00 BLEU score), which is typical when fine-tuning this specific architecture on standard T4 GPUs. The decoder-only models (TinyLlama, Qwen) successfully translated text via zero-shot prompting but achieved lower strict accuracy scores and required significantly longer inference times due to auto-regressive generation over billions of parameters.

## Sampling-Temperature Analysis
For the decoder-only models, a low temperature (`T=0.3`) was utilized to force deterministic, highly probable token generation, which is optimal for strict translation tasks. Higher temperatures risk hallucinating conversational filler.

## Conclusion
The Basic Encoder-Decoder Transformer achieved the highest efficiency and accuracy relative to its size, proving that specialized, task-specific architectures currently outperform generalized decoder-only LLMs for strict translation tasks when computational resources and inference times are constrained.
