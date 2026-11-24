# Neural Machine Translation with LSTM and Bahdanau Attention (English to German)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Assignment 3: Neural Machine Translation with LSTM and Attention Mechanisms**  
> *Natural Language Processing — Fall 2026, Alexandria National University*  
> **Author**: Mohamed Baz ([@FekyBaz](https://github.com/FekyBaz))

---

## 📌 Project Overview

This repository provides a clean, fully reproducible PyTorch implementation comparing two foundational Sequence-to-Sequence (Seq2Seq) architectures for English-to-German Neural Machine Translation:
1. **Baseline Seq2Seq (LSTM without Attention)**: An Encoder LSTM compresses the variable-length source sentence into a single fixed-length context vector (final hidden and cell states $(h_T, c_T)$), which initializes the Decoder LSTM to generate target words autoregressively.
2. **Seq2Seq with Bahdanau Attention**: A Bidirectional Encoder LSTM produces annotations for every source time step. At each decoding step, an additive attention mechanism computes dynamic alignment weights ($\alpha_{t,s}$) over all source tokens, allowing direct information flow and resolving the fixed-length information bottleneck.

---

## 🏗️ Architecture & Mathematical Formulation

### 1. Baseline Seq2Seq (Sutskever et al., 2014)
* **Encoder**:
  $$e_s = \text{Embedding}(x_s)$$
  $$(h_s^{enc}, c_s^{enc}) = \text{LSTM}(e_s, (h_{s-1}^{enc}, c_{s-1}^{enc}))$$
* **Context Vector**:
  $$h_0^{dec} = h_S^{enc}, \quad c_0^{dec} = c_S^{enc}$$
* **Decoder**:
  $$(h_t^{dec}, c_t^{dec}) = \text{LSTM}(\text{Embedding}(y_{t-1}), (h_{t-1}^{dec}, c_{t-1}^{dec}))$$
  $$P(y_t \mid y_{<t}, x) = \text{softmax}(W_o h_t^{dec} + b_o)$$

### 2. Seq2Seq with Bahdanau Additive Attention (Bahdanau et al., 2014)
* **Bidirectional Encoder**:
  $$H_s = [\vec{h}_s; \overleftarrow{h}_s] \in \mathbb{R}^{2 \cdot d_{enc}}$$
  $$s_0 = \tanh(W_{init} [\vec{h}_S; \overleftarrow{h}_1] + b_{init})$$
* **Bahdanau Attention Alignment Energy**:
  $$e_{t, s} = v_a^\top \tanh(W_a s_{t-1} + U_a H_s)$$
* **Normalized Attention Weights ($\alpha_{t,s}$)**:
  $$\alpha_{t, s} = \frac{\exp(e_{t, s})}{\sum_{k=1}^S \exp(e_{t, k})}$$
  *(Padded source positions are masked with $-\infty$ prior to softmax).*
* **Dynamic Context Vector ($c_t$)**:
  $$c_t = \sum_{s=1}^S \alpha_{t, s} H_s$$
* **Decoder Step & Prediction**:
  $$(s_t, c_t^{dec}) = \text{LSTMCell}([e_t; c_t], (s_{t-1}, c_{t-1}^{dec}))$$
  $$P(y_t \mid y_{<t}, x) = \text{softmax}(W_o [s_t; c_t; e_t] + b_o)$$

---

## 📁 Repository Structure

```
nlp-en-de-seq2seq-attention/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Ignores large checkpoints and temp files
├── train_all.py                   # Complete automated training & evaluation script
├── src/
│   ├── __init__.py                # Module exports
│   ├── data.py                    # Normalization, German preservation, Vocab, DataLoader
│   ├── models_baseline.py         # Encoder, Decoder, Seq2Seq without attention
│   ├── models_attention.py        # BiLSTM Encoder, Bahdanau Attention, Attention Decoder
│   ├── train.py                   # Epoch loops, teacher forcing, loss evaluation, checkpointing
│   ├── evaluate.py                # Greedy decoding, BLEU 1-4, ROUGE-L, length bucketing
│   └── visualize.py               # Loss curve plotting, 2D attention heatmaps
├── notebooks/
│   └── en_de_seq2seq_attention.ipynb  # Self-contained, runnable Jupyter notebook
├── outputs/
│   ├── checkpoints/               # Best saved model weights (.pt)
│   ├── figures/                   # Loss curves, length analysis, 2D attention heatmaps
│   └── results/                   # Evaluation metrics JSON, length analysis, error table
└── report/
    ├── report.md                  # Comprehensive academic report
    └── report.pdf                 # Formatted PDF submission report
```

---

## ⚡ Quickstart & Reproducibility

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/FekyBaz/nlp-en-de-seq2seq-attention.git
cd nlp-en-de-seq2seq-attention
pip install -r requirements.txt
```

### 2. End-to-End Training & Evaluation
To train both models, plot loss curves, extract attention weights, and evaluate BLEU/ROUGE:
```bash
python train_all.py
```

### 3. Interactive Notebook
Launch Jupyter to explore step-by-step:
```bash
jupyter notebook notebooks/en_de_seq2seq_attention.ipynb
```

---

## 📊 Experimental Results

### Quantitative Metrics (Held-Out Test Set)

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Seq2Seq (LSTM)** | 33.99 | 20.60 | 13.27 | 8.41 | 38.47 |
| **Seq2Seq + Bahdanau Attention** | **43.86** | **30.30** | **21.61** | **15.37** | **49.46** |
| *Absolute Gain ($\Delta$)* | *+9.87* | *+9.70* | *+8.34* | *+6.96* | *+10.99* |

### Information Bottleneck Mitigation across Sentence Lengths

| Length Bucket | Test Count | Baseline BLEU-4 | Attention BLEU-4 | Baseline ROUGE-L | Attention ROUGE-L | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short (3–6 words)** | 5 | 15.69 | **26.14** | 63.73 | **66.61** | Context fits readily in cell state |
| **Medium (7–11 words)** | 220 | 10.75 | **20.51** | 42.34 | **54.01** | Attention tracks descriptive modifiers |
| **Long (12–18+ words)** | 275 | 7.28 | **13.00** | 34.92 | **45.52** | **Bottleneck resolved by dynamic alignment** |

---

## 🔍 Attention Heatmaps & Qualitative Error Analysis

The model extracts the full 2D attention matrix $\alpha_{t,s}$ during inference.
- **Monotonic Diagonal**: Captures direct sequential correspondences (e.g., subject-verb pairings).
- **Off-Diagonal Alignments**: Reflects grammatical reordering (e.g., English adjective-noun order vs. German compound nouns, and German verb-final subordinate clauses).
- Generated figures are stored in `outputs/figures/attention_sample_*.png`.

---

## 📜 Academic Integrity
This project was developed for **Assignment 3 (NLP - Fall 2026)** at **Alexandria National University**. All code, models, and analytical write-ups are original and reproducible.
