# 🔍 Fake News Detection Using Transformer Models and Explainable AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?logo=huggingface)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab)
![License](https://img.shields.io/badge/License-Academic-green)
![Course](https://img.shields.io/badge/CECS%20551-CSULB-red)

> A complete, reproducible fake news detection pipeline combining fine-tuned **DistilBERT** with **LIME** and **SHAP** Explainable AI — achieving an **F1-score of 0.9995** and **perfect precision of 1.0000** on 4,404 held-out test articles.

---

## 👥 Team — Gradient Descenters



**Course:** CECS 551 — Artificial Intelligence | California State University, Long Beach  
**Instructor:** Dr. Benyamin Haghi

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9927 | 0.9901 | 0.9948 | 0.9925 |
| TF-IDF + Linear SVM | 0.9975 | 0.9967 | 0.9981 | 0.9974 |
| **DistilBERT (Fine-tuned)** ⭐ | **0.9995** | **1.0000** | **0.9991** | **0.9995** |

> ✅ DistilBERT achieves **perfect precision** — every article predicted as Real was genuinely Real, with **zero false positives** and only **2 misclassifications** out of 4,404 test articles.

---

## 📁 Repository Structure

```
fake-news-detection/
│
├── README.md                                          ← You are here
├── Fake_News_Detection_GradientDescenters.ipynb       ← Main notebook (all 11 sections)
└── report/
    └── IEEE_FakeNewsDetection_GradientDescenters.pdf  ← Final IEEE conference report
```

> ⚠️ Dataset files (`Fake.csv`, `True.csv`) are **not included** due to file size. See Step 3 below for download instructions.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Fake and Real News Dataset (Bisaillon)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| Raw size | 44,898 articles |
| After preprocessing | 44,034 articles |
| Fake articles | 23,481 (52.3%) — Label 0 |
| Real articles | 21,417 (47.7%) — Label 1 |
| Class balance | Near-balanced — no resampling required |
| Time period | 2015–2017 US political news |
| Columns | `title`, `text`, `subject`, `date`, `label` |

> ⚠️ **Domain limitation:** This dataset covers only 2015–2017 US political news. The trained model may not generalize to other domains, languages, or time periods.

---

## ⚙️ Pipeline Overview

The notebook is organized into **11 sequential sections**:

```
Fake.csv + True.csv  (Kaggle Download)
         ↓
Section 3  — Data Loading
         ↓
Section 4  — EDA (class distribution, text length, subject categories)
         ↓
Section 5  — Preprocessing (deduplication, cleaning, 500-word truncation)
         ↓
Section 6  — Train / Validation / Test Split (80% / 10% / 10%, stratified, seed=42)
         ↓
Section 7  — Evaluation Helper Functions
         ↓
Section 8  — Baseline Models (TF-IDF + Logistic Regression | TF-IDF + Linear SVM)
         ↓
Section 9  — DistilBERT Fine-Tuning (3 epochs, AdamW, linear warmup)
         ↓
Section 10 — Model Comparison (all 3 models, all 4 metrics)
         ↓
Section 11 — Explainable AI (LIME word-level + SHAP token-level explanations)
```

---

## 🚀 How to Run (Step-by-Step)

### Step 1 — Open the Notebook in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Go to [colab.research.google.com](https://colab.research.google.com/) → **File → Upload notebook** → select `Fake_News_Detection_GradientDescenters.ipynb`

---

### Step 2 — Set Runtime to GPU ⚠️ Required

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

> GPU is **required** for DistilBERT training. Running on CPU will be extremely slow (~8–10x longer).

---

### Step 3 — Download the Dataset from Kaggle

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Click **Download** (requires free Kaggle account)
3. Extract the zip — you need two files:
   - `Fake.csv`
   - `True.csv`

---

### Step 4 — Upload Dataset Files to Colab

In Colab, click the 📁 **folder icon** in the left sidebar → then click the **upload icon** → upload both `Fake.csv` and `True.csv`

> Files must be in the Colab root directory (`/content/`) — this is where the notebook expects them.

---

### Step 5 — Run All Cells in Order

```
Runtime → Run all
```

Or run cells **one by one from top to bottom** — do not skip any cell, as each section depends on the ones above.

> ⏱️ **Estimated total runtime: ~30–35 minutes on T4 GPU**

| Section | Estimated Time |
|---|---|
| Sections 1–7 (setup, EDA, baselines) | ~3–5 minutes |
| Section 9 (DistilBERT training, 3 epochs) | ~20–25 minutes |
| Section 11 (LIME + SHAP explanations) | ~5–8 minutes |

---

## 📦 Dependencies

All dependencies are **automatically installed in Section 1** (Cell 1) of the notebook. No manual setup needed in Colab.

```python
!pip install transformers datasets lime shap --quiet
```

| Package | Purpose |
|---|---|
| `transformers` | DistilBERT model, tokenizer, and linear warmup scheduler (HuggingFace) |
| `datasets` | HuggingFace dataset utilities |
| `torch` | PyTorch deep learning backend (pre-installed in Colab) |
| `scikit-learn` | TF-IDF vectorizer, Logistic Regression, Linear SVM, evaluation metrics (pre-installed) |
| `lime` | Local Interpretable Model-Agnostic Explanations |
| `shap` | SHapley Additive exPlanations |
| `pandas` | Data loading and manipulation (pre-installed) |
| `numpy` | Numerical operations (pre-installed) |
| `matplotlib` | Plotting and visualizations (pre-installed) |
| `seaborn` | Statistical visualizations (pre-installed) |

> Only `transformers`, `datasets`, `lime`, and `shap` need to be installed — everything else comes pre-installed in Google Colab.

---

## 🔧 Hyperparameters

### DistilBERT Fine-Tuning

| Parameter | Value | Justification |
|---|---|---|
| Base model | `distilbert-base-uncased` | 40% smaller, 60% faster than BERT — retains 97% performance |
| Trainable parameters | 66,955,010 | Full end-to-end fine-tuning |
| Max token length | 256 | Balances context coverage and GPU memory |
| Batch size | 16 | Fits within T4 GPU VRAM |
| Epochs | 3 | Sufficient convergence without overfitting |
| Learning rate | 2e-5 | Standard for transformer fine-tuning |
| Optimizer | AdamW | `weight_decay=0.01` for L2 regularization |
| Scheduler | Linear warmup | 660 warmup steps out of 6,606 total (10%) |
| Gradient clipping | Max norm 1.0 | Prevents exploding gradients |
| Random seed | 42 | Fixed across numpy, PyTorch, and CUDA for reproducibility |
| Best checkpoint | Saved by validation F1 | Prevents overfitting to training set |

### TF-IDF Baselines

| Parameter | Value |
|---|---|
| Max features | 50,000 |
| N-gram range | (1, 2) — unigrams and bigrams |
| TF scaling | Sublinear (`sublinear_tf=True`) |
| Min document frequency | 2 |
| LR regularization C | 1.0, max iterations 1,000 |
| SVM regularization C | 1.0, max iterations 2,000 |

### Data Splits

| Split | Proportion | Articles |
|---|---|---|
| Train | 80% | 35,227 |
| Validation | 10% | 4,403 |
| Test | 10% | 4,404 |

> All splits use **stratified sampling** (`stratify=y`) with `random_state=42` to preserve class distribution across all sets.

---

## 🧠 Explainable AI (XAI) — Section 11

We apply two complementary post-hoc XAI methods to understand individual DistilBERT predictions:

### LIME — Local Interpretable Model-Agnostic Explanations
- Creates **300 perturbed versions** of each article by randomly masking words
- Fits a local linear surrogate model to approximate the DistilBERT decision boundary
- Output: bar chart showing which words push toward **Fake (red)** or **Real (green)**

### SHAP — SHapley Additive exPlanations
- Uses the **Partition Explainer** with Shapley values from cooperative game theory
- Assigns mathematically grounded **token-level** contribution scores
- Output: highlighted text showing token-level fake/real signal strength

### Both methods were applied to:
- ✅ One correctly predicted **Fake** article (99.56% confidence)
- ✅ One correctly predicted **Real** article (100.00% confidence)

### Key Finding
> The model learned that **Reuters wire service datelines** (tokens like `"reuters"`, `"nairobi"`) strongly signal Real news, while **sensationalist political language** (tokens like `"candidate"`, `"america"`, `"fits"`) signals Fake news. This reveals a partial reliance on **source attribution patterns** rather than pure content-level features — a vulnerability future work should address.

---

## ⚠️ Known Limitations

| Limitation | Impact |
|---|---|
| Dataset covers only **2015–2017 US political news** | May not generalize to other domains, languages, or time periods |
| Model partially relies on **Reuters source signals** | Vulnerable to adversaries mimicking wire service formatting |
| Evaluated on a **single dataset** | Cross-dataset testing on LIAR and FakeNewsNet is future work |
| LIME uses **300 perturbation samples** (recommended: 500+) | Explanation stability may vary between runs |
| Only article **body text** is used | Title, subject, and date features are not exploited |
| Hyperparameters **not exhaustively tuned** | Compute constraints on Colab limited grid search |
| No **k-fold cross-validation** | Fixed seed split may not fully represent model variance |

---

## 📚 Key References

| Reference | Contribution |
|---|---|
| Devlin et al. (2019) | BERT — bidirectional transformer pre-training |
| Sanh et al. (2019) | DistilBERT — 40% smaller, 60% faster, 97% of BERT performance |
| Ribeiro et al. (2016) | LIME — local surrogate model explanations |
| Lundberg & Lee (2017) | SHAP — Shapley value attribution framework |
| Kaliyar et al. (2021) | FakeBERT — BERT applied to fake news detection |
| Wang (2017) | LIAR — benchmark dataset for fake news detection |
| Bisaillon (2020) | Kaggle Fake-and-Real-News dataset used in this project |

Full IEEE-formatted reference list (18 references) is available in the project report.

---

## 📄 Report

The full IEEE-style conference report is in `report/IEEE_FakeNewsDetection_GradientDescenters.pdf`.

Sections: Abstract · Introduction · Literature Review (6 subsections, 17 citations) · Methods (7 subsections) · Results (4 subsections with tables and figures) · Discussion · Conclusion & Future Work · References

---

## 📝 License

This project was submitted as part of **CECS 551 — Artificial Intelligence** at California State University, Long Beach. For academic use only.
