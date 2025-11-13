# Transformer Architecture Implementation (PyTorch)

**Course:** DAM202 [Year3-Sem1]  
**Practical:** Practical 6  

---

## Table of Contents

- [Overview](#overview)
- [Architecture Components](#architecture-components)
- [Implementation Details](#implementation-details)
- [Hyperparameters](#hyperparameters)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Key Features](#key-features)
- [Architecture Diagram](#architecture-diagram)
- [Mathematical Formulations](#mathematical-formulations)
- [References](#references)

---

## Overview

This project implements the complete **Transformer architecture** from scratch using PyTorch, following the seminal paper "Attention Is All You Need" by Vaswani et al. (2017). The implementation is modular, well-documented, and adheres to the original paper's specifications for the **Base Model configuration**.

### Objective

To gain a deep, practical understanding of the Transformer architecture by implementing all core components including:
- Multi-Head Attention mechanism
- Positional Encoding
- Encoder-Decoder framework
- Feed-Forward Networks
- Masking mechanisms

---

## Architecture Components

### 1. **ScaledDotProductAttention**
- Implements the core attention mechanism
- Computes attention scores with scaling factor: `scores = (Q @ K^T) / √d_k`
- Applies masking and dropout
- Returns context vectors and attention weights

### 2. **MyMultiHeadAttention**
- Splits input into `h=8` parallel attention heads
- Each head has dimension `d_k = d_v = 64`
- Applies linear projections: `W_Q`, `W_K`, `W_V`
- Concatenates heads and applies final projection `W_O`
- Includes dropout for regularization

### 3. **MyPositionwiseFFN**
- Two-layer feed-forward network
- Architecture: `Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)`
- Applied identically to each position
- Provides non-linearity and transformation capacity

### 4. **MyPositionalEncoding**
- Fixed sinusoidal positional encodings
- Formula:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
- Added to input embeddings
- Enables model to utilize sequence order

### 5. **MyEncoderLayer**
- Single encoder block containing:
  - Multi-Head Self-Attention sublayer
  - Position-wise Feed-Forward Network
  - Residual connections around each sublayer
  - Layer Normalization after each sublayer

### 6. **MyDecoderLayer**
- Single decoder block containing:
  - Masked Multi-Head Self-Attention sublayer
  - Multi-Head Cross-Attention sublayer (attends to encoder output)
  - Position-wise Feed-Forward Network
  - Residual connections and Layer Normalization

### 7. **MyEncoder**
- Stack of `N=6` identical encoder layers
- Processes source sequence
- Outputs contextual representations

### 8. **MyDecoder**
- Stack of `N=6` identical decoder layers
- Processes target sequence
- Attends to encoder output via cross-attention

### 9. **MyTransformer**
- Complete end-to-end model
- Integrates encoder and decoder stacks
- Includes embedding layers and output projection
- Xavier uniform parameter initialization

---

## Implementation Details

### Masking Mechanisms

#### 1. **Padding Mask**
```python
def create_padding_mask(seq, pad_token=0):
    """
    Creates mask to hide <PAD> tokens
    - Shape: (B, 1, 1, seq_len)
    - 1 for real tokens, 0 for padding
    - Applied in encoder & decoder self-attention
    - Applied in cross-attention
    """
```

#### 2. **Look-Ahead (Causal) Mask**
```python
def create_look_ahead_mask(size):
    """
    Prevents attention to future positions
    - Shape: (1, size, size)
    - Upper triangular mask
    - Applied in decoder self-attention only
    """
```

#### 3. **Combined Mask**
```python
def combine_masks(pad_mask_src=None, pad_mask_tgt=None, look_ahead_mask=None):
    """
    Combines padding and look-ahead masks
    - Used for decoder self-attention
    - Shape: (B, 1, seq_len, seq_len)
    """
```

### Residual Connections & Layer Normalization

Each sublayer follows the pattern:
```
Output = LayerNorm(x + Sublayer(x))
```

Where:
- `x` is the input to the sublayer
- `Sublayer(x)` is the output of the sublayer (attention or FFN)
- Dropout is applied after each sublayer
- Layer Normalization uses `eps=1e-6`

---

## Hyperparameters

### Base Model Configuration

| Parameter | Notation | Value | Description |
|-----------|----------|-------|-------------|
| **Model Dimension** | `d_model` | **512** | Size of embeddings and layer outputs |
| **Number of Layers** | `N` | **6** | Layers in encoder and decoder stacks |
| **Number of Heads** | `h` | **8** | Parallel attention heads |
| **Key/Value Dimension** | `d_k`, `d_v` | **64** | Per-head dimension (d_model / h) |
| **Feed-Forward Dimension** | `d_ff` | **2048** | Inner dimension of FFN |
| **Dropout Rate** | `p_dropout` | **0.1** | Applied to sublayers and embeddings |
| **Max Sequence Length** | `max_seq_len` | **512** | Maximum sequence length supported |

---

## Installation & Requirements

### Prerequisites

```bash
Python 3.7+
PyTorch 1.7+
```

### Dependencies

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd transformer-implementation

# Install PyTorch (if not already installed)
pip install torch

# Or using conda
conda install pytorch -c pytorch
```

---

## Usage

### Basic Example

```python
# Import the model
from Practical6 import MyTransformer, create_padding_mask, create_look_ahead_mask, combine_masks

# Define hyperparameters
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1
src_vocab_size = 1000
tgt_vocab_size = 1000

# Instantiate the model
model = MyTransformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=512,
    dropout=dropout,
)

# Create dummy input
B = 2  # Batch size
src_seq_len = 10
tgt_seq_len = 9

src_input = torch.randint(1, src_vocab_size, (B, src_seq_len))
tgt_input = torch.randint(1, tgt_vocab_size, (B, tgt_seq_len))

# Create masks
src_pad_mask = create_padding_mask(src_input, pad_token=0)
tgt_pad_mask = create_padding_mask(tgt_input, pad_token=0)
look_ahead = create_look_ahead_mask(tgt_seq_len)
tgt_mask = combine_masks(pad_mask_tgt=tgt_pad_mask, look_ahead_mask=look_ahead)

# Forward pass
logits = model(
    src_input, 
    tgt_input, 
    src_mask=src_pad_mask.squeeze(2).squeeze(2), 
    tgt_mask=tgt_mask, 
    memory_mask=src_pad_mask
)

print(f"Output shape: {logits.shape}")  # (B, tgt_seq_len, tgt_vocab_size)
```

### Running the Test Script

```python
# Run the built-in smoke test
if __name__ == '__main__':
    # The notebook includes a complete test at the end
    # Expected output:
    # src_input shape: torch.Size([2, 10])
    # tgt_input shape: torch.Size([2, 9])
    # src_pad_mask shape: torch.Size([2, 1, 1, 10])
    # tgt_mask shape: torch.Size([2, 1, 9, 9])
    # logits shape: torch.Size([2, 9, 1000])
    # Successful forward pass with Base Model hyperparameters!
```

---

## Code Structure

```
transformer-implementation/
│
├── Practical6.ipynb          # Main implementation notebook
├── practical6.md                 # This file
├── architecture.md # Visual architecture diagram
│
└── Components:
    ├── ScaledDotProductAttention    # Core attention mechanism
    ├── MyMultiHeadAttention         # Multi-head attention module
    ├── MyPositionwiseFFN            # Feed-forward network
    ├── MyPositionalEncoding         # Positional encoding
    ├── MyEncoderLayer               # Single encoder layer
    ├── MyDecoderLayer               # Single decoder layer
    ├── MyEncoder                    # Encoder stack (N=6 layers)
    ├── MyDecoder                    # Decoder stack (N=6 layers)
    ├── MyTransformer                # Complete transformer model
    └── Masking utilities            # Helper functions for masks
```

---

## Key Features

### 1. **Modular Design**
- Each component is a separate `nn.Module`
- Easy to understand, modify, and extend
- Clear separation of concerns

### 2. **Complete Implementation**
- All components from the original paper
- Proper masking mechanisms
- Correct dimensional handling
- Xavier uniform initialization

### 3. **Well-Documented Code**
- Detailed docstrings for each class
- Inline comments explaining key operations
- Type hints for function parameters

### 4. **Proper Masking**
- Padding mask for ignoring `<PAD>` tokens
- Look-ahead mask for preventing future attention
- Correct mask broadcasting and combination

### 5. **Dimensional Correctness**
- All tensor shapes properly managed
- Correct reshaping for multi-head attention
- Proper broadcasting for attention scores

### 6. **Dropout Regularization**
- Applied to attention weights
- Applied to sublayer outputs
- Applied to positional encodings

---

## Architecture Diagram

The architecture diagram visually maps the theoretical Transformer architecture to the specific PyTorch implementation:

### Components Shown:
- **Input Processing**: Embeddings and positional encoding
- **Encoder Stack**: 6 layers of self-attention and FFN
- **Decoder Stack**: 6 layers with masked self-attention, cross-attention, and FFN
- **Output Layer**: Linear projection to vocabulary size
- **Tensor Dimensions**: Annotated at each stage
- **Masking**: Padding and look-ahead masks illustrated
- **Cross-Attention**: Connection from encoder to decoder


[Click here to view Architecture Diagram](architecture.pdf)
---

## Mathematical Formulations

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- `Q`: Query matrix (seq_len_q, d_k)
- `K`: Key matrix (seq_len_k, d_k)
- `V`: Value matrix (seq_len_v, d_v)
- `d_k`: Dimension of keys (64 in base model)

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Where:
- `W_i^Q ∈ R^(d_model × d_k)`
- `W_i^K ∈ R^(d_model × d_k)`
- `W_i^V ∈ R^(d_model × d_v)`
- `W^O ∈ R^(h·d_v × d_model)`

### Position-wise Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Where:
- `W_1 ∈ R^(d_model × d_ff)` with `d_ff = 2048`
- `W_2 ∈ R^(d_ff × d_model)`

### Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos`: Position in the sequence
- `i`: Dimension index

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

Where:
- `μ`: Mean across d_model dimension
- `σ²`: Variance across d_model dimension
- `ε = 1e-6`: Small constant for numerical stability
- `γ, β`: Learnable parameters

---

## Implementation Highlights

### 1. **Attention Masking**
```python
# In ScaledDotProductAttention.forward()
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-1e9'))
```

### 2. **Multi-Head Splitting**
```python
# In MyMultiHeadAttention.forward()
Q = Q.view(B, -1, self.h, self.d_k).transpose(1, 2)  # (B, h, seq, d_k)
```

### 3. **Residual Connection + LayerNorm**
```python
# In MyEncoderLayer.forward()
attn_output, _ = self.self_attn(x, x, x, mask=src_mask)
x = self.norm1(x + self.dropout(attn_output))
```

### 4. **Embedding Scaling**
```python
# In MyTransformer.forward()
src_emb = self.src_tok_emb(src_input) * math.sqrt(self.d_model)
```

### 5. **Xavier Initialization**
```python
# In MyTransformer._init_parameters()
for p in self.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

---

## Testing

### Smoke Test Results

The implementation includes a comprehensive smoke test that verifies:

- Correct tensor shapes throughout the forward pass
- Proper masking application
- Dimensional consistency
- Successful forward pass with base model hyperparameters

### Expected Output:
```
src_input shape: torch.Size([2, 10])
tgt_input shape: torch.Size([2, 9])
src_pad_mask shape: torch.Size([2, 1, 1, 10])
tgt_mask shape: torch.Size([2, 1, 9, 9])
logits shape: torch.Size([2, 9, 1000])
Successful forward pass with Base Model hyperparameters!
```

---

## Key Concepts Explained

### 1. **Why Multi-Head Attention?**
- Allows the model to attend to information from different representation subspaces
- Each head can learn to focus on different aspects of the input
- Analogous to having multiple convolutional filters in CNNs

### 2. **Why Positional Encoding?**
- Transformers have no inherent notion of sequence order
- Positional encodings inject information about token positions
- Sinusoidal functions allow the model to generalize to longer sequences

### 3. **Why Masking?**
- **Padding Mask**: Prevents attention to meaningless padding tokens
- **Look-Ahead Mask**: Ensures autoregressive property in decoder (can't see future)
- Both are essential for correct model behavior

### 4. **Why Residual Connections?**
- Helps with gradient flow during backpropagation
- Allows training of very deep networks
- Provides identity mapping shortcut

### 5. **Why Layer Normalization?**
- Stabilizes training
- Reduces internal covariate shift
- Applied after residual connections

---

## Learning Outcomes

After implementing this project, you will understand:

1. **Attention Mechanisms**: How scaled dot-product attention works mathematically and computationally
2. **Multi-Head Attention**: Why and how parallel attention heads improve model capacity
3. **Positional Information**: How transformers encode sequence order without recurrence
4. **Encoder-Decoder Architecture**: How the two stacks work together for sequence-to-sequence tasks
5. **Masking Strategies**: The critical role of padding and causal masks
6. **Residual Learning**: How skip connections enable deep network training
7. **Layer Normalization**: Its role in stabilizing transformer training

---

## References

### Primary Reference
- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).** 
  *Attention is all you need.* 
  Advances in neural information processing systems, 30.
  - [Paper Link](https://arxiv.org/abs/1706.03762)

### Additional Resources
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [Attention Is All You Need - Yannic Kilcher](https://www.youtube.com/watch?v=iDulhoQ2pro)

---

## Notes

### Design Decisions

1. **Post-Layer Normalization**: Applied after residual connection (x + Sublayer(x))
2. **Xavier Initialization**: Used for all parameters with dimension > 1
3. **Dropout Placement**: Applied to attention weights and sublayer outputs
4. **Mask Convention**: 1 for allowed positions, 0 for masked positions (inverted in attention)




