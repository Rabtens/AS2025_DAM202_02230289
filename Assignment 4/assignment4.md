# Transformer Decoder for Sequence Generation

**Module:** DAM202  
**Assignment:** Assignment 4 — Transformer Decoder   
**Date:** 22 November 2025

---

## 1. Project Overview

This project develops a Transformer Decoder–based sequence generation system. The goal is to understand how decoder blocks work, how autoregressive generation functions, and how different decoding strategies affect generated text.

### The notebook includes:

- A full Encoder–Decoder Transformer model
- Causal-masked Decoder for autoregressive generation
- Training on a small sequence-to-sequence dataset
- Implemented decoding strategies:
  - Greedy Search
  - Beam Search
  - Nucleus Sampling (Top-p)
- Model training, text generation, evaluation, and analysis

This project follows the assignment requirements from DAM202.

---

## 2. Learning Objectives

The notebook delivers the following objectives:

### Understand Transformer decoder mechanism

- Implemented Multi-Head Attention
- Causal masking for autoregressive prediction
- Decoder stack with feed-forward networks

### Implement autoregressive text generation

- Token-by-token prediction
- Prevents seeing future tokens using causal mask

### Compare different decoding strategies

- Greedy
- Beam search
- Nucleus sampling

### Train and evaluate an encoder–decoder model

- Mini dataset prepared
- Train → Validate → Generate
- Evaluate using simple metrics and sample outputs

### Analyze generation quality

- Compare outputs from 3 decoding methods
- Discussion provided in analysis section

---

## 3. Project Architecture

Below is the architecture used in the notebook.

```
                   ┌──────────────────────────────┐
                   │           Encoder             │
                   │  (N layers, self-attention)   │
                   └───────────────┬──────────────┘
                                   │
                                   ▼
                      Encoder Output Memory
                                   │
                                   ▼
              ┌───────────────────────────────┐
              │            Decoder             │
              │ (Masked self-attn + cross-attn)│
              └─────────────────┬──────────────┘
                                │
                                ▼
                     Linear Layer + Softmax
                                │
                                ▼
                       Token Predictions
```

### Encoder

- Embedding
- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Network

### Decoder

- Masked Self-Attention (causal mask)
- Cross-Attention (attends to encoder output)
- Feed Forward Network

### Autoregressive Generation

- Start with `<BOS>` token
- Predict next token
- Append
- Continue until `<EOS>` token

---

## 4. Dataset Description

The project uses a small synthetic or simplified sequence-to-sequence dataset created inside the notebook.

### Examples include:

- English → Simplified Dzongkha phrases
- English → English pattern continuation
- Short instructional sequences

This keeps training fast while demonstrating the architecture properly.

---

## 5. Training Process

The training loop includes:

- Input/Target preparation
- Encoder forward pass
- Decoder forward pass
- Teacher forcing
- CrossEntropy Loss
- Adam optimizer
- Padding masks + Causal masks

Model checkpoints are saved after training.

### Training outputs include:

- Loss curve
- Sample predictions
- Attention scores

---

## 6. Decoding Strategies Implemented

### 1. Greedy Search

- Always choose the most probable next token
- Fast but sometimes low quality
- May repeat tokens

### 2. Beam Search

- Keeps multiple "best possible paths"
- More strategic
- Better output quality
- Slower

### 3. Nucleus Sampling (Top-p)

- Chooses from the smallest probability mass ≥ p
- Produces more natural and creative text
- Less deterministic

All three strategies were tested in the notebook.

---

## 7. Evaluation & Results

The notebook evaluates:

### Output samples

Model-generated sentences under each decoding method.

### Comparison

- **Greedy** → Most stable but dry
- **Beam** → Most accurate
- **Top-p** → More natural, creative

### Attention visualizations

Shows which words the decoder attends to.

### Loss Curve

Training behavior shown graphically.

---

## 8. Analysis of Results

### Greedy Search

- Very deterministic
- Good for predictable outputs
- Often repetitive

### Beam Search

- Best for translation tasks
- Produces meaningful sequences
- Balances variety + accuracy

### Nucleus Sampling

- Human-like text generation
- Good for open-ended tasks
- Slight randomness

![alt text](<Screenshot from 2025-11-22 18-11-30.png>)

![alt text](<Screenshot from 2025-11-22 18-11-40.png>)

### Overall

The Transformer Decoder works correctly for sequence generation, and the decoding strategies clearly affect the quality of the produced sequences.

---

## 9. Files Included

| File                          | Description                                          |
|-------------------------------|------------------------------------------------------|
| Assignment4.ipynb    | Full implementation of encoder–decoder Transformer   |
| model_checkpoint.pt           | Saved PyTorch weights (if generated)                 |
| assignment4.md                     | This report                                          |

---

## 10. How to Run the Notebook

### Run in Google Colab:

- Upload the notebook to Colab
- Set runtime to GPU
- Run cells from top to bottom
- Generated text will appear at the bottom

### Requirements

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`

---

## 11. Conclusion

This assignment successfully implements:

- A full Transformer Encoder–Decoder
- Autoregressive generation
- Causal masking
- Three decoding algorithms
- Model training and evaluation
- Analysis of generated outputs

---

## 12. References

- Vaswani et al., 2017 — *"Attention is All You Need"*
- PyTorch Transformer Documentation
- NLP Sequence-to-Sequence tutorials