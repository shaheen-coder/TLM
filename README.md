# 🧠 TinyLM v3: Decoder-Only Architecture Research Report

<p align="center">
  <img src="https://img.shields.io/badge/Status-Failed_Research_v3-red?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-blue?style=for-the-badge" alt="Arch">
  <img src="https://img.shields.io/badge/Positional_Encoding-RoPE-purple?style=for-the-badge" alt="RoPE">
  <img src="https://img.shields.io/badge/Framework-TensorFlow/Keras-orange?style=for-the-badge" alt="Framework">
  <img src="https://img.shields.io/badge/GPU-Tesla_T4-green?style=for-the-badge" alt="GPU">
</p>

---

## 📌 Project Overview

This is the **third iteration of TinyLM**, continuing the decoder-only Transformer architecture introduced in v2 with **Rotary Positional Embeddings (RoPE)**. The key change in this iteration was a **significant dataset scale-up** — from ~460K tokens to approximately **10M tokens total** — directly addressing the primary bottleneck identified in v2.

Training followed a two-phase strategy:
- **Phase 1 — Foundational Pretraining:** `fd_dataset` (~9M tokens of story/narrative text), 5 epochs.
- **Phase 2 — Fine-tuning:** `ft_dataset` (~1M tokens of chat dialogue), 40 epochs with a reduced learning rate.

Despite the dramatic increase in data volume, inference still exhibits failure modes — repetitive outputs, template-locked responses, and poor generalisation on unseen prompts. This report documents the architecture, training setup, inference results, and an updated root cause analysis.

> [!CAUTION]
> **Key Finding:** Scaling the dataset from ~460K to ~10M tokens produced measurable training loss improvements but did not resolve inference failures. The model remains unable to generalise across prompt variations. The bottleneck has shifted from raw data volume toward **data domain mismatch**, **fine-tuning distribution drift**, and **model capacity**.

---

## 🏗️ Architecture Design

TinyLM v3 retains the same decoder-only Transformer design as v2 — a GPT-style architecture with weight-tied embeddings and RoPE positional encoding.

| Component | Specification |
| :--- | :--- |
| **Embedding** | Token Embedding, `d_model=128`, `vocab_size=10,840`, `mask_zero=True` |
| **Positional Encoding** | Rotary Positional Embedding (RoPE), `max_wavelength=10,000` |
| **Attention** | Multi-Head Self-Attention, `num_heads=4`, `key_dim=32` (`d_model / num_heads`) |
| **Feed-Forward Network** | Two Dense layers: `dff=256 → d_model=128`, ReLU activation |
| **Normalization** | LayerNorm (`ε=1e-6`) after attention and FFN (Post-LN) |
| **Residual Connections** | Add layers after attention and FFN |
| **LM Head** | Weight-tied projection to vocab (`embedding.embeddings ᵀ`) + trainable bias |
| **Dropout** | `dropout=0.2` (attention + embedding) |
| **Num Layers** | Configurable (single Transformer block in this run) |

### Causal Masking

A combined **causal + padding mask** is applied during self-attention:

- **Causal mask** — lower-triangular boolean matrix preventing future token leakage (`tf.linalg.band_part`).
- **Padding mask** — derived from `embedding.compute_mask` to suppress `[PAD]` token influence.
- Both masks are `AND`-ed and passed to `MultiHeadAttention`. Padded positions are also zeroed in the hidden state after the embedding layer (`x *= padding_mask`).

---

## ⚙️ Training Setup

### Datasets

| Dataset | Purpose | Approx. Tokens | Batch Size | Epochs |
| :--- | :--- | :--- | :--- | :--- |
| `fd_dataset` | Foundational pretraining (story data) | ~9M | 64 | 5 |
| `ft_dataset` | Fine-tuning (chat dialogue) | ~1M | 12 | 40 |
| `val_dataset` | Validation (chat dialogue) | — | 12 | — |

### Hyperparameters

| Parameter | Phase 1 (Pretrain) | Phase 2 (Fine-tune) |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | `1e-4` | `2e-5` |
| **Weight Decay** | `1e-4` | `1e-4` |
| **β₁ / β₂ / ε** | `0.9 / 0.999 / 1e-7` | `0.9 / 0.999 / 1e-7` |
| **Loss** | Masked Sparse Categorical Crossentropy | Masked Sparse Categorical Crossentropy |
| **Data Shuffling** | Buffer 10,000 | Buffer 10,000 |
| **Prefetching** | `tf.data.AUTOTUNE` | `tf.data.AUTOTUNE` |
| **Logging** | TensorBoard (histogram, per epoch) | TensorBoard (histogram, per epoch) |

---

## 📉 Training Logs

### Phase 1 — Foundational Pretraining (`fd_dataset`, 5 epochs)

| Epoch | Loss | Masked Accuracy |
| :---: | :---: | :---: |
| 1 | 6.3731 | 0.1380 |
| 2 | 5.4212 | 0.1898 |
| 3 | 4.9790 | 0.2096 |
| 4 | 4.6881 | 0.2220 |
| 5 | 4.4785 | 0.2313 |

Training loss dropped from **6.37 → 4.48** over 5 epochs (~30% reduction), indicating the model is learning surface-level language patterns from the foundational corpus.

### Phase 2 — Fine-tuning (`ft_dataset`, 40 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 7.3404 | 0.1181 | 7.6115 | 0.0910 |
| 5 | 5.7015 | 0.2193 | 7.1119 | 0.1061 |
| 10 | 5.1862 | 0.2622 | 6.9664 | 0.1080 |
| 15 | 4.8790 | 0.2862 | 6.8749 | 0.1267 |
| 20 | 4.6436 | 0.3066 | 6.8495 | 0.1279 |
| 25 | 4.4581 | 0.3221 | 6.8436 | 0.1270 |
| 30 | 4.3125 | 0.3379 | 6.8014 | 0.1262 |
| 35 | 4.1820 | 0.3507 | 6.7858 | 0.1295 |
| 40 | 4.0661 | 0.3632 | 6.7723 | 0.1280 |

> [!WARNING]
> **Critical Observation:** Training loss steadily falls from 7.34 → 4.07, while validation loss only improves marginally from 7.61 → 6.77 and plateaus after ~epoch 10. Validation accuracy stalls at ~12–13% throughout fine-tuning. This gap indicates the model is memorising fine-tune training patterns rather than learning generalisable chat behaviour. The pretrain distribution (story text) and fine-tune distribution (chat dialogue) are substantially different, causing the model to essentially re-learn from scratch in Phase 2 — explaining the validation loss spike at fine-tune epoch 1 (7.61) versus the end of pretraining (4.48).

---

## 🤖 Inference Case Studies

Inference was evaluated across three decoding strategies. All prompts are in Tamil-English code-mixed (Tanglish) style, directed at the model persona "shaAI".

### 1. Greedy Decoding

| Prompt | Response |
| :--- | :--- |
| `vanakkam shaAI, epdi irukka?` | `Aama da , naan oru small model .` |
| `hello shaAI` | `naan oru small model . LSTM layers -> -> -> -> -> -> -> -> -> -> [repeating]` |
| `hi shaAI, good morning` | `Aama da , naan oru program .` |
| `hey shaAI` | `naan oru small model . LSTM layers layers layers layers .` |
| `shaAI, unga peru epdi varum? nee yaaru shaAI?` | `naan oru small model . but naan oru small model . but naan oru small model . but naan ' ' ' ' [repeating]` |

### 2. Top-p Sampling (p=0.7)

| Prompt | Response |
| :--- | :--- |
| `vanakkam shaAI, epdi irukka?` | `sure ! naan oru ID . en peru ' sha ' ll always .` |
| `hello shaAI` | `aama , naan oru AI . but data analysis la enaku konjam struggle aagum .` |
| `hi shaAI, good morning` | `sure ! naan oru limit irukka ? [AI] Rombha nalla irukken .` |
| `hey shaAI` | `naan oru small model . LSTM layers layers layers vachu input pannanga .` |
| `shaAI, unga peru epdi varum? nee yaaru shaAI?` | `actually , naan empathetic a respond pannanga . naan oru neural network . en peru ' ' Tiny padichurukken !` |

### 3. Top-p Sampling (p=0.5)

| Prompt | Response |
| :--- | :--- |
| `vanakkam shaAI, epdi irukka?` | `well , naan oru program . but naan oru neural network use panren . but blue color pathi neraya kelvi pattrukken !` |
| `hello shaAI` | `naan oru AI model . LSTM layers -> -> -> -> -> LSTM layers -> -> -> -> -> dense layers -> -> [repeating]` |
| `hi shaAI, good morning` | `Aama da , naan oru AI .` |
| `hey shaAI` | `naan oru small model . LSTM layers layers layers . fewer parameters .` |
| `shaAI, unga peru epdi varum? nee yaaru shaAI?` | `well , naan oru AI ' ' functioning ' panrathu illa .` |

> [!WARNING]
> **Identified Issues:**
> - Responses collapse to `"naan oru small model"` / `"naan oru program"` regardless of prompt.
> - Greedy decoding produces severe repetition loops (`->`, `layers layers layers`, repeated quotes).
> - The model leaks fine-tune training artifacts — mentioning LSTM layers, dense layers — suggesting it memorised meta-descriptions of itself from the training data rather than learning genuine conversational patterns.
> - Sampling helps break repetition loops slightly but introduces incoherence.
> - No meaningful prompt conditioning: semantically distinct prompts receive near-identical responses.

---

## 🔬 Root Cause Analysis

| Aspect | Observation |
| :--- | :--- |
| **Architecture** | Decoder-only Transformer with RoPE remains sound. No design-level flaw identified. |
| **Causal Masking** | Correctly implemented; combined causal + padding mask. |
| **Weight Tying** | LM head shares embedding weights — appropriate for a small model. |
| **Pretraining Data** | ~9M tokens of story text is substantially better than before, but the domain (narrative prose) is distant from Tanglish chat dialogue. |
| **Fine-tuning Distribution Shift** | The jump from story prose → chat dialogue causes a sharp train/val loss spike at FT epoch 1, suggesting the pretraining representations are not transferring cleanly. |
| **Fine-tuning Data Volume** | ~1M tokens of chat data spread over 40 epochs causes rapid memorisation of high-frequency templates. |
| **Validation Plateau** | Val accuracy stalls at ~12–13% from epoch 10 onward — indicating the model has reached the ceiling of what this fine-tune data can teach it. |
| **Model Capacity** | A single Transformer block with `d_model=128` has very limited representational depth. Cannot form complex conditional distributions. |
| **Template Collapse** | The phrase `"naan oru small model"` appears to be a dominant training sample or was over-represented in fine-tune data; the model defaults to it under all prompts. |
| **Tokenizer** | The 10,840-token vocabulary is likely word-level and poorly suited to Tanglish morphology and code-mixing. |
| **Repetition** | No repetition penalty in decoding; greedy mode amplifies degenerate loops. |

---

## 📊 Version Comparison

| Feature | TinyLM v1 (LSTM-Attention) | TinyLM v2 (Decoder-Only) | TinyLM v3 (Decoder-Only + Scale) |
| :--- | :--- | :--- | :--- |
| **Core Unit** | LSTM Encoder-Decoder | Self-Attention Transformer | Self-Attention Transformer |
| **Positional Encoding** | Implicit (LSTM order) | RoPE | RoPE |
| **Pretraining Data** | None | None | ~9M tokens (fd_dataset) |
| **Fine-tune Data** | ~460K tokens | ~460K tokens | ~1M tokens (ft_dataset) |
| **Final Train Loss** | 2.96 | — | 4.07 (FT) |
| **Final Val Loss** | 9.55 | — | 6.77 (FT) |
| **Inference Quality** | Poor | Marginally better | Similar failure modes; worse repetition in greedy |
| **Repetition in Greedy** | Present | Present | More severe |

---

## 📎 Future Directions

Scale alone did not resolve the failure. The next iteration must address **data domain alignment**, **model capacity**, and **decoding quality**.

### 🗂️ Dataset

- [ ] **Domain-aligned pretraining:** Source Tanglish pretraining data from Tamil Twitter/X threads, YouTube comments, WhatsApp-style chat corpora — not general story text. Domain proximity matters more than raw token count.
- [ ] **Reduce template dominance:** Audit fine-tune data for over-represented response templates (e.g., `"naan oru small model"`). Balance or deduplicate heavily repeated samples.
- [ ] **Multi-turn dialogue structure:** Ensure fine-tune data contains diverse multi-turn exchanges, not just single Q&A pairs.
- [ ] **Back-translation augmentation:** Translate Tamil sentences → English → back to generate additional Tanglish variation.
- [ ] **Curate for register diversity:** Cover greetings, factual Q&A, emotional registers, follow-up questions — not a single conversational mode.

### 🔤 Tokenizer

- [ ] **Train a domain-specific BPE or SentencePiece tokenizer** on Tanglish corpus — subword tokenization handles code-mixed morphology and English loanwords far better than word-level.
- [ ] **Target vocabulary size:** Experiment in the 8K–16K subword range trained specifically on Tanglish data.
- [ ] **Evaluate token fertility** (tokens per word) before and after — lower fertility on domain text indicates a better-fit vocabulary.

### 🏗️ Architecture & Training

- [ ] **Increase model depth:** Move to 2–4 Transformer blocks and `d_model=256` or `d_model=512` to improve representational capacity.
- [ ] **Warm-up + cosine LR schedule:** Replace fixed LR with a warm-up + cosine decay schedule for more stable fine-tuning.
- [ ] **Repetition penalty at inference:** Apply log-probability penalties for recently generated tokens to suppress degenerate loops.
- [ ] **EarlyStopping on val loss:** Stop fine-tuning when val loss stops improving to prevent over-memorisation.
- [ ] **Gradient clipping:** Add `clipnorm` to AdamW to stabilise training on the chat distribution.
- [ ] **Pre-norm (Pre-LN) instead of Post-LN:** Pre-LN is more stable for small models and shallow networks.

---

### Author
**Shaheen** *Machine Learning Engineer* [![GitHub](https://img.shields.io/badge/GitHub-shaheen--coder-181717?style=flat&logo=github)](https://github.com/shaheen-coder)

---
