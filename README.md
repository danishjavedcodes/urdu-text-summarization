# Extractive Summarization of Urdu Language

Official implementation companion to the IEEE conference paper [**Extractive Summarization of Urdu Language using Deep Learning Techniques on a Custom Dataset**](https://ieeexplore.ieee.org/document/10935058) (ICET 2024).

This repository provides reproducible experiments for **extractive Urdu text summarization** using a supervised sentence-classification pipeline. Two deep learning approaches are evaluated: a **bidirectional LSTM** model and a **customized BERT** encoder with a classification head. Summaries are produced by selecting sentences predicted as salient, and quality is measured with **ROUGE** scores.

---

## Overview

Urdu is a morphologically rich, low-resource language with limited annotated corpora for summarization. This work addresses that gap by:

1. Introducing **UCES-v1** (*Urdu Corpus for Extractive Summarization*), a supervised dataset of Urdu documents paired with short reference summaries across multiple domains.
2. Training and comparing **LSTM** and **customized BERT** models for sentence-level extractive summarization.
3. Reporting systematic ROUGE-based evaluation and analysis of each model’s strengths on Urdu text.

The LSTM model leverages sequential processing suited to sentence-level decisions. The customized BERT model uses pre-trained contextual embeddings and attention, and generalizes well to held-out documents.

---

## Paper

| | |
|---|---|
| **Title** | Extractive Summarization of Urdu Language using Deep Learning Techniques on a Custom Dataset |
| **Authors** | Danish Javed, Maham Shehzadi, Musadaq Mansoor, Sarah Iqbal |
| **Venue** | 2024 19th International Conference on Emerging Technologies (ICET) |
| **DOI** | [10.1109/icet63392.2024.10935058](https://doi.org/10.1109/icet63392.2024.10935058) |
| **IEEE Xplore** | [Document 10935058](https://ieeexplore.ieee.org/document/10935058/) |

### Citation

```bibtex
@inproceedings{javed2024extractive,
  title={Extractive Summarization of Urdu Language using Deep Learning Techniques on a Custom Dataset},
  author={Javed, Danish and Shehzadi, Maham and Mansoor, Musadaq and Iqbal, Sarah},
  booktitle={2024 19th International Conference on Emerging Technologies (ICET)},
  pages={1--6},
  year={2024},
  doi={10.1109/icet63392.2024.10935058}
}
```

---

## Repository Structure

```
Extractive/
├── README.md                          # This file
├── IEEE Xplore Full-Text PDF.png      # Paper reference (IEEE Xplore)
└── urdu-text-summarization/           # Implementation
    ├── lstm.ipynb                     # Bidirectional LSTM pipeline
    ├── bert.ipynb                     # Customized BERT pipeline
    ├── d.ipynb                        # Combined development notebook
    ├── rouge_scores.py                # ROUGE evaluation utilities
    └── public/                        # Static assets (fonts)
```

---

## Dataset: UCES-v1

The **UCES-v1** dataset contains Urdu source documents with human-authored short summaries, organized for supervised extractive learning.

| Column | Description |
|--------|-------------|
| `Text` | Full Urdu document |
| `Type` | Domain label (e.g. History, Health, Nature, General) |
| `Short Summary` | Reference summary used to derive sentence-level labels |

**Download:** [UCES-v1 on Kaggle](https://www.kaggle.com/dsv/8540609)

After downloading, place the preprocessed pickle file as `data.pkl` in the `urdu-text-summarization/` directory (see [Setup](#setup)).

---

## Models

### 1. Bidirectional LSTM

- Sentence tokenization with **spaCy** (`ur` blank pipeline).
- Keras `Tokenizer` + padded sequences; binary labels from overlap with reference summary sentences.
- Architecture: `Embedding` → `Bidirectional(LSTM(64))` → `Dropout` → `Dense(sigmoid)`.
- Trained with Adam and binary cross-entropy; saved as `LSTM_Summarizer.h5`.

### 2. Customized BERT (extractive)

- Built on **`bert-base-uncased`** via TensorFlow (`TFBertModel`).
- Token-level classification head for extractive sentence selection.
- Fine-tuned with attention masks; checkpoints saved under `BERT MODIFIED/`.

Both models follow the same high-level flow: **tokenize → label sentences → train classifier → select top sentences → evaluate with ROUGE**.

---

## Setup

### Requirements

- Python 3.10+
- Jupyter Notebook or JupyterLab
- GPU recommended for BERT fine-tuning (optional for LSTM)

Install dependencies:

```bash
cd urdu-text-summarization
pip install tensorflow transformers torch pandas numpy scikit-learn spacy rouge-score
python -m spacy download ur_core_news_sm   # optional; notebooks use spacy.blank('ur')
```

### Data preparation

1. Download **UCES-v1** from [Kaggle](https://www.kaggle.com/dsv/8540609).
2. Export or convert to a pandas-compatible pickle named `data.pkl` with columns `Text`, `Type`, and `Short Summary`.
3. Place `data.pkl` in `urdu-text-summarization/`.

---

## Usage

Run notebooks from the `urdu-text-summarization/` directory:

| Notebook | Purpose |
|----------|---------|
| `lstm.ipynb` | End-to-end LSTM training, inference, and ROUGE evaluation |
| `bert.ipynb` | BERT-based extractive model training and evaluation |
| `d.ipynb` | Development notebook combining LSTM and BERT workflows |

**ROUGE evaluation** (shared utility):

```bash
python rouge_scores.py
```

The helper in `rouge_scores.py` computes mean ROUGE-1, ROUGE-2, and ROUGE-L F-measures over reference/hypothesis pairs.

---

## Evaluation

Summaries are evaluated with the [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) metric family (ROUGE-1, ROUGE-2, ROUGE-L), following standard extractive summarization practice. Run the evaluation cells at the end of each notebook after generating predictions on the test split.

---

## Key Findings (from the paper)

- The **bidirectional LSTM** model performs strongly on Urdu text thanks to its sequential processing of sentence representations.
- The **customized BERT** model benefits from pre-trained embeddings and attention, producing quality summaries on unseen documents.
- The **UCES-v1** dataset enables supervised learning for Urdu extractive summarization and supports future research in low-resource NLP.

---

## Future Work

Directions identified in the paper include:

- Exploring additional neural architectures and hyperparameter tuning.
- Incorporating advanced attention mechanisms for Urdu summarization.
- Expanding UCES-v1 with more diverse topics and writing styles.
- Domain-specific customization and end-to-end model optimization.

---

## Authors

- **Danish Javed** — GIK Institute, FCSE, Topi, KPK
- **Maham Shehzadi** — GIK Institute, FCSE, Topi, KPK
- **Musadaq Mansoor** — GIK Institute, FCSE, Topi, KPK
- **Sarah Iqbal** — GIK Institute, FCSE, Topi, KPK

---

## Related Work

This project builds on and cites prior Urdu summarization research, including extractive and abstractive methods. Notable related resources:

- [End-to-End Urdu Abstractive Text Summarization with Dataset and Improvement in Evaluation Metric](https://doi.org/10.1109/ACCESS.2024.3377463) (Raza & Shahzad, *IEEE Access*, 2024)
- [Extractive Text Summarization Models for Urdu Language](https://doi.org/10.1016/j.ipm.2020.102383) (Nawaz et al., 2020)
- [Meeting the Challenge: A Benchmark Corpus for Automated Urdu Meeting Summarization](https://doi.org/10.1016/j.ipm.2024.103734) (Sadia et al., 2024)

---

## License

Source code in this repository is provided for research and reproducibility. Please cite the ICET 2024 paper when using this code or the UCES-v1 dataset. Dataset licensing follows the terms on the [Kaggle dataset page](https://www.kaggle.com/dsv/8540609).

---

## Acknowledgments

Published in proceedings of **ICET 2024**. Dataset hosting via **Kaggle** (UCES-v1). Pre-trained weights from [Hugging Face Transformers](https://huggingface.co/google-bert/bert-base-uncased).
