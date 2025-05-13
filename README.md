# Token-Driven VDB Reasoning for Robust Sentiment Classification

This repository contains research code and experiments for enhancing video database (VDB)-based classification using token-level attention tuning. The project focuses on hybrid reasoning, where a pretrained BERT tokenizer and attention module guide examples into a VDB-driven classifier for sentiment analysis on SST-2.

---

## 🔍 Project Goal

Improve the classification of semantically ambiguous or misclassified samples by:

- Identifying latent trigger tokens in error zones
- Learning attention patterns to emphasize informative parts of the input
- Shifting the representation space toward more correct semantic neighborhoods in the VDB

---

## 🧠 Core Components

- **Baseline (tokreg_v1)**: Token-level attention regularization with frozen BERT
- **Contrastive Attention**: Fine-tuning attention heads on token-centered contrastive sets extracted from error zones
- **Combined Models**: Integration of trained attention heads with downstream classifiers
- **VDB Reasoning**: Embedding-based retrieval and decision via nearest neighbors
- **Error-Aware Loop**: Selective correction based on attention-adjusted embeddings

---

## 📊 Evaluation Summary

| Model Variant                            | Training Epochs | Accuracy | Macro F1 | Notes                                      |
|------------------------------------------|------------------|----------|----------|--------------------------------------------|
| `tokreg_v1`                              | 3                | 0.8704   | 0.8703   | Token-level attention regularization       |
| `tokreg_v1_finetuned`                    | 3 + 5            | 0.8727   | 0.8727   | +5 epochs classifier fine-tuning           |
| `tokreg_contrastive`                     | 3 + 5            | 0.8899   | 0.8898   | Contrastive attention + classifier tuning  |

> Experimental data available by request.
---

## 🔧 Architecture Overview

- `bert-base-uncased`: frozen transformer for tokenization and embeddings
- `AttentionEncoder`: trainable attention layer for token importance weighting
- `ContrastiveAttentionEncoder`: trained on token-conditioned triplets from error examples
- `CombinedAttentionModel`: uses contrastive attention + classifier layer
- `evaluate_model()`: wrapper that performs VDB encoding + kNN classification

---

## 🧪 Key Findings

- VDB-based reasoning is highly sensitive to local embedding structure
- Attention alone must significantly shift vector location to change VDB output
- Contrastive-tuned attention improves accuracy by ~2% over baseline
- Most test errors remain located near ambiguous or weak-density embedding zones
- Dense clusters dominate inference; sparse zones show higher VDB instability

---

## 🚧 Next Directions

- Visualize attention token migration across epochs
- Integrate LLM-generated rephrasings (teacher guidance) for error zones
- Cluster VDB into adaptive or weighted prototypes
- Analyze classification delta on borderline and low-density samples

---

## 📜 License

MIT License — free for academic and research use.

---

## 🤝 Contributions

Open to ideas and collaborations around hybrid neural-retrieval reasoning, explainable AI, and attention dynamics.

```