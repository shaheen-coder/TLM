🧠 TinyLM Failure Analysis (Research Report)

📌 Overview

This project implements a small-scale language model (TinyLM) using an LSTM-based encoder-decoder architecture with Luong-style attention.
The model was trained on a relatively small dataset (~460K tokens) for conversational text generation.

Despite achieving decreasing training loss and moderate training accuracy, the model fails to generalize during inference and produces low-quality, repetitive, and incoherent outputs.

This document provides a complete technical analysis of:

- Architecture
- Training behavior
- Loss curves
- Inference outputs
- Root causes of failure

---

🏗️ Architecture

Model Design

Input → Embedding → Encoder LSTM → Decoder LSTM → Attention → Projection → Output logits

Components

- Shared Embedding
  
  - "vocab_size = 10840"
  - "d_model = 256"

- Encoder
  
  - LSTM (returns sequences + states)

- Decoder
  
  - LSTM initialized with encoder states

- Attention
  
  - Luong-style dot-product attention ("tf.keras.layers.Attention")

- Output Layer
  
  - Weight tying with embedding matrix
  - Learned bias

---

📊 Training Summary

Dataset

- Total tokens: ~460,000
- Vocabulary size: 10,840
- Validation : ~10%

---

Training Metrics

Metric| Final Value
Train Loss| ~2.96
Validation Loss| ~9.55
Train Accuracy| ~50%
Validation Accuracy| ~11%

---

📉 Loss Behavior

Key Pattern

- Training loss decreases steadily
- Validation loss increases continuously

This indicates:

«⚠️ Severe overfitting»

---

📈 Training Loss Graph

"Training vs Validation Loss"
![losses] (docs/traning.jpg)

Interpretation

- Early epochs: validation loss < training loss
- Later epochs: divergence increases sharply

👉 Model memorizes training data instead of learning general patterns

---

🔬 Evalution Analysis

Evaluation Losses

![eval loss] (docs/eval_loss.jpg)

Observations

- Poor clustering of semantically related tokens
- No meaningful structure in embedding space
- High overlap between unrelated tokens

👉 Indicates weak semantic learning

---

🤖 Inference Results

Greedy Decoding

Prompt  : hello shaAI
Response: actually , enna , naan oru seyya poreenga .

Prompt  : hi shaAI, good morning
Response: actually , naan oru nalla irukken . but naan layers la irukken .

---

Top-k Sampling

Prompt  : vanakkam shaAI, epdi irukka?
Response: naan vachirundha nalla irukken . but usually irukkinga la irukku !

---

Observed Issues

- Repetitive phrases: ""actually", "naan oru""
- Broken grammar
- Weak prompt conditioning
- Mixed and incoherent responses

---


🧠 Key Insight

«✅ Low loss ≠ good language model
❌ High accuracy ≠ understanding»

The model learns:

- Token-level transitions

But fails at:

- Context understanding
- Coherent generation

---

🔄 Failure Pattern Summary

Aspect| Behavior
Training| Successful
Validation| Diverges
Inference| Poor quality
Generalization| Failed

---

📌 Conclusion

TinyLM demonstrates a classic failure case where:

- Training appears successful
- Metrics look reasonable
- But inference quality is poor

This highlights the importance of:

- Matching architecture to task
- Using sufficient data
- Evaluating beyond training metrics

---

📎 Future Work

- Implement GPT-style decoder-only model
- Visualize attention weights
- Analyze embedding similarity using cosine distance
- Scale dataset and compare performance

---

Author:
Shaheen ( @shaheen-coder ) (ML Engineer)

---
