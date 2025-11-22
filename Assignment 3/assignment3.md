# DAM202 — Transformer Encoder Practical Report

## Sentiment Classification on IMDB Dataset
### Using From-Scratch Transformer Encoder & BERT Fine-Tuning

---

## 1. Introduction

This practical focuses on building and comparing two NLP models for movie review sentiment classification:

- A **Transformer Encoder** built completely from scratch in PyTorch
- A **fine-tuned BERT-base** model from HuggingFace

The goal was to understand how Transformer architectures work internally and compare them to pre-trained language models.

The **IMDB 50K Movie Review** dataset was used. Reviews are labelled as positive or negative.

---

## 2. Project Objectives

- Understand data preprocessing, tokenization, and exploratory data analysis (EDA)
- Implement Transformer encoder components:
  - Positional Encoding
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Feed-Forward Network
  - Encoder Blocks
- Train the model using:
  - Mixed precision
  - One-cycle learning rate scheduler
  - AdamW optimizer
- Fine-tune BERT-base-uncased for the same task
- Visualize attention maps and compare both models
- Report training performance and evaluation metrics

---

## 3. Dataset Description

**Dataset:** IMDB 50K Movie Reviews

**Labels:**
- positive
- negative

### 3.1 Class Distribution

*(Insert the bar chart from notebook Cell 5)*

- **Balanced dataset:** 25,000 positive & 25,000 negative

### 3.2 Text Length Distribution

*(Insert histogram from Cell 5)*

- Most reviews contain **50–300 words**

### 3.3 Sample Reviews

**Example positive review:**
> "An amazing film with brilliant acting and a touching story…"

**Example negative review:**
> "This is one of the worst movies ever made…"

---

## 4. Data Splits

A stratified split was applied to maintain balanced labels.

| Split      | Percentage | Size   |
|------------|------------|--------|
| Train      | 60%        | 30,000 |
| Validation | 20%        | 10,000 |
| Test       | 20%        | 10,000 |

---

## 5. Tokenization

**Tokenizer used:** BERT Tokenizer (bert-base-uncased)
- **Padding:** max_length = 128
- **Truncation:** enabled

### Token Stats

- Average tokens per review: approx. **180**
- Max tokens found: about **5000** (long reviews)
- Truncated to **128 tokens** for training speed
- Tokenized data saved in `.pt` files for fast loading

---

## 6. Transformer Encoder Architecture (From Scratch)

The Transformer Encoder implemented includes:

### 6.1 Embedding Layer
- Converts token IDs → vectors of size `d_model` (e.g., 512)

### 6.2 Positional Encoding
- Adds sine/cosine patterns to embed sequence order

### 6.3 Multi-Head Self-Attention
- Splits embeddings into multiple attention heads
- Computes:

```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

### 6.4 Feed-Forward Network
- 2-layer MLP with GELU activation

### 6.5 Residual Connections + LayerNorm
- Crucial for stable deep training

### 6.6 Stacked Encoder Layers
- **6 layers total**

### 6.7 Classification Head
- Global average pool → Linear → Softmax

---

## 7. Architecture Diagram 

```
Tokens → Embedding → Add Positional Encoding
       ↓
┌───────────── Transformer Encoder Layer (x6) ─────────────┐
│ Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm │
└───────────────────────────────────────────────────────────────┘
       ↓
Mean Pooling
       ↓
Dropout → Linear Layer → Output (Positive/Negative)
```

---

## 8. Training Configuration (Scratch Model)

| Hyperparameter   | Value      |
|------------------|------------|
| d_model          | 512        |
| n_heads          | 8          |
| FFN dim          | 2048       |
| Layers           | 6          |
| Batch size       | 32         |
| Optimizer        | AdamW      |
| Learning rate    | 3e-4       |
| Scheduler        | One-Cycle  |
| Mixed Precision  | Yes        |
| Epochs           | 5          |

---

## 9. Training Results (Scratch Model)

### 9.1 Validation Performance

Average across epochs:

| Metric   | Score |
|----------|-------|
| Accuracy | ~0.80 |
| F1-score | ~0.79 |

### 9.2 Test Performance

Final test results:
- **Accuracy:** ~0.80
- **F1:** ~0.79

The model learns reasonably well but is weaker than BERT due to no pretraining.

---

## 10. BERT Fine-Tuning Setup

**BERT-base-uncased** provides:
- 12 attention layers
- 768-dimensional hidden states
- Pre-trained on 3.3B words

### Layerwise Learning Rate Decay
- Lower layers = lower LR
- Higher layers = higher LR
- This prevents catastrophic forgetting

---

## 11. Training Configuration (BERT)

| Hyperparameter   | Value           |
|------------------|-----------------|
| Base LR          | 2e-5            |
| Batch size       | 16              |
| Epochs           | 3               |
| Warmup           | 100 steps       |
| Mixed Precision  | Yes             |
| Param Grouping   | Layerwise LR    |

---

## 12. BERT Training Results

BERT performs significantly better.

### Validation F1
- F1-score reaches **0.93+** by epoch 3

### Test Results
- **Accuracy:** 0.93–0.94
- **F1-score:** 0.93–0.94

**Pretrained models clearly outperform scratch implementations.**

---

## 13. Attention Visualization

Both models' attention weights can be visualized.

### Common Observations
- Middle layers focus strongly on keywords like "good", "bad", "boring", "amazing"
- Higher layers focus on sentiment-heavy phrases
- Scratch encoder attention is more noisy
- BERT shows cleaner and more meaningful attention

![alt text](<Screenshot from 2025-11-22 13-44-53.png>)

---

## 14. Ablation Study

To study the effect of model components, various experiments were tried:

### Ablations
- **Remove positional encoding** → accuracy drops by ~10%
- **Reduce number of heads from 8 → 2** → performance drops ~7%
- **Remove LayerNorm** → model becomes unstable
- **Reduce d_model** → smaller models learn slower

---

## 15. Comparison Summary

| Model                | Test Accuracy | Test F1 | Training Time |
|----------------------|---------------|---------|---------------|
| Scratch Transformer  | ~0.80         | ~0.79   | Long          |
| BERT Fine-Tuned      | ~0.94         | ~0.94   | Short         |

### Conclusion
- Pretrained BERT is far superior
- Scratch model is useful for learning but not for real-world performance

---

## 16. Final Conclusion

This practical helped understand how Transformer Encoders work internally. The from-scratch model shows the mechanics of attention, positional encoding, and encoder layers. However, the fine-tuned BERT model demonstrates the power of transfer learning.

### Key Takeaways:
- Transformers require large pretraining to perform well
- BERT fine-tuning is efficient and accurate
- Mixed-precision training speeds up training on GPU
- Attention visualization helps understand model behavior
- Building models from scratch improves conceptual clarity

---

## 17. References

- Vaswani et al., *Attention Is All You Need*, 2017
- Devlin et al., *BERT: Pretraining of Deep Bidirectional Transformers*, 2018
- HuggingFace Transformers
- IMDB Movie Review Dataset (Kaggle)