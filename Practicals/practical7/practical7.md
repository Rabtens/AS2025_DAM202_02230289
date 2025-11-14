# Multi-Task Learning: NER and QA with Shared Transformer

## 1. Project Overview

This practical focuses on:

- Building a shared Transformer encoder
- Creating two task-specific heads (NER head + QA head)
- Preparing toy datasets for testing (offline, no internet needed)
- Training a multi-task model using a custom loop
- Visualizing loss curves and evaluating predictions

Everything is written in simple PyTorch so it is easy to understand and run anywhere.

---

## 2. Multi-Task Learning Idea (Simple Explanation)

Instead of training two separate models:

- One for NER
- One for QA

…we train one model with:

- **Shared Encoder**: Learns general language understanding
- **NER Head**: Learns token classification
- **QA Head**: Learns to find answer start and end positions

This allows the tasks to share knowledge and reduce overfitting.

---

## 3. Architecture Diagram

Below is the architecture used in the notebook — kept simple and easy to read.

```
                      +-----------------------+
                      |       Input Text      |
                      +-----------+-----------+
                                  |
                                  v
                +-----------------------------------------+
                |       Shared Transformer Encoder        |
                |  (multi-layer self-attention blocks)    |
                +------------+----------------------------+
                             |             
              +--------------+--------------+
              |                             |
              v                             v
      +---------------+             +----------------+
      |   NER Head    |             |    QA Head     |
      | (Linear Layer)|             | (Start & End   |
      |               |             |   Linear Layer) |
      +-------+-------+             +--------+--------+
              |                               |
              v                               v
     +----------------+              +---------------------+
     |  NER Output    |              |  QA Output          |
     | (tag per token)|              |(start/end positions)|
     +----------------+              +---------------------+
```

[clickheretoviewthearchitecture](architecture.pdf)

---

## 4. Dataset (Toy Offline Dataset)

To ensure the notebook runs without internet, small synthetic datasets are used:

- **Toy NER dataset**: Simple sentences with manually created token labels
- **Toy QA dataset**: Small contexts, questions, and short answer spans

These keep the model light and fast to train, while still demonstrating the full pipeline.

---

## 5. Tokenization

A simple whitespace tokenizer is used in this practical:

- Splits sentences into tokens
- Maps tokens to integer IDs
- Pads sequences to the same length

This avoids downloading HuggingFace tokenizers.

---

## 6. Model Components

### 6.1 Shared Encoder

A tiny Transformer with:

- Multi-head self-attention
- Feed-forward layers
- Positional encoding

### 6.2 NER Head

A single linear layer that outputs:

- `[num_tokens × num_labels]`

### 6.3 QA Head

Two linear layers:

- **Start Logits**: `num_tokens`
- **End Logits**: `num_tokens`

---

## 7. Multi-Task Training

During training:

- Each batch is either NER or QA (alternating)
- For NER batches → compute NER loss
- For QA batches → compute QA loss
- Total loss = sum of both task losses
- Backpropagation updates shared encoder + both heads

### Loss Functions

- **NER**: Cross-Entropy loss per token
- **QA**: Cross-Entropy loss for start and end positions

---

## 8. Evaluation

### NER Evaluation

- Compare predicted tag IDs with true labels
- Simple accuracy is used (because dataset is small)

### QA Evaluation

- Check whether predicted start/end tokens align with true answer
- Measure span accuracy

![alt text](<Screenshot from 2025-11-14 12-05-15.png>)

---

## 9. Visualisation

The notebook plots:

- Total Loss
- NER Loss
- QA Loss

This helps verify the training stability and task balance.

![alt text](<Screenshot from 2025-11-14 12-04-58.png>)

---

## 10. Key Learning Outcomes

From this practical, you learn:

- How Transformers can handle multiple tasks together
- How a shared encoder helps tasks support each other
- How to design multi-task models in PyTorch from scratch
- How to build custom training loops (because HuggingFace Trainer doesn't support MTL easily)
- How to visualise loss and evaluate predictions

---

## 11. File Structure

```
├── practical7.ipynb   
├── practical7.md                         
└── screenshot_images                        
```

---

## 12. Conclusion

This practical shows a complete, simple, and working example of multi-task NLP learning using Transformers. Even with small toy datasets, the structure matches real multi-task systems used in modern NLP research.

The model trains fast, is easy to understand, and demonstrates the full pipeline of:

- Data preparation
- Tokenization
- Model building
- Multi-task loss
- Training loop
- Evaluation
- Plotting results

