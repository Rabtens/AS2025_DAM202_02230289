# Translation Model – Practical Report

## 1. Introduction

This practical focuses on building a Neural Machine Translation (NMT) model using TensorFlow. The goal is to translate sentences from one language to another (for example, from English to a target language) using deep learning methods.

The notebook includes all the steps from installing required libraries to building, training, and testing the translation model.

---

## 2. Objective

The main objective of this practical is to:

- Understand how machine translation works using sequence-to-sequence (Seq2Seq) models.
- Learn to preprocess text data for translation tasks.
- Implement and train a translation model using TensorFlow.
- Evaluate and test the translation model with example inputs.

---

## 3. Required Libraries

The following libraries were used in the project:
```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
import time
from sklearn.model_selection import train_test_split
import unicodedata
```

---

## 4. Dataset

The dataset used in this notebook consists of paired sentences (source and target language). Each sentence pair was cleaned and normalized to remove unwanted characters, punctuation, and spacing errors.

---

## 5. Data Preprocessing

Steps involved in preprocessing:

- Loading and reading the dataset.
- Cleaning text (removing special symbols and accents).
- Tokenizing sentences into words.
- Padding sequences to make them equal length.
- Splitting the data into training and testing sets.

---

## 6. Model Architecture

The translation model is based on the Encoder–Decoder (Seq2Seq) architecture:

- **Encoder:** Converts the input sentence into a context vector.
- **Decoder:** Uses the context vector to generate the translated sentence word by word.
- **Attention mechanism:** Helps the model focus on important words during translation.

---

## 7. Training

The model is trained using the Adam optimizer and SparseCategoricalCrossentropy loss function. Training involves several epochs until the model learns accurate word-to-word mapping between input and target languages.

A progress bar and loss curve were displayed to monitor model improvement.

---

## 8. Evaluation and Testing

After training, the model was tested with sample sentences to evaluate its accuracy. Predicted translations were compared with the actual target translations to observe the model's performance.

---

## 9. Results and Observations

- The model successfully translated short sentences.
- Accuracy improved with more training epochs.
- Attention visualization showed how the model focused on key words during translation.

---

## 10. Conclusion

This practical helped in understanding how Neural Machine Translation works using Seq2Seq models with Attention in TensorFlow. It also demonstrated the importance of text preprocessing and model tuning for better translation quality.

---

## 11. Screenshot / Output

 Model training result or translation output.

 ![alt text](<Screenshot from 2025-10-22 12-21-38.png>)

 ![alt text](<Screenshot from 2025-10-22 12-21-50.png>)

 ![alt text](<Screenshot from 2025-10-22 12-22-04.png>)

 ![alt text](<Screenshot from 2025-10-22 12-22-24.png>)

 ![alt text](<Screenshot from 2025-10-22 12-22-31.png>)

 ![alt text](<Screenshot from 2025-10-22 12-22-50.png>)
---

## 12. References

- TensorFlow Documentation: https://www.tensorflow.org
- Sequence to Sequence Learning with Neural Networks (Google, 2014)
- Attention Mechanism in NMT (Bahdanau et al., 2015)