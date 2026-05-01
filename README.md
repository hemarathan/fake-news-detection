# Fake News Detection Using Transformer Models and Explainable AI

**Team:** Gradient Descenters  
**Members:** Hemarathan Munnangi, Mauli Arunkumar Pandya, Mansi Bharatbhai Lathiya  
**Course:** CECS 551 — Artificial Intelligence | California State University, Long Beach  

---

## Project Overview
This project implements a complete fake news detection pipeline using Natural Language Processing (NLP) and Explainable AI (XAI). News articles are classified as **Fake (0)** or **Real (1)** using traditional machine learning baselines and a fine-tuned DistilBERT transformer model.

---

## Results Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9927 | 0.9901 | 0.9948 | 0.9925 |
| TF-IDF + Linear SVM | 0.9975 | 0.9967 | 0.9981 | 0.9974 |
| **DistilBERT (Fine-tuned)** | **0.9995** | **1.0000** | **0.9991** | **0.9995** |

---

## Dataset
- **Source:** [Kaggle — Fake and Real News Dataset (Bisaillon)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size:** 44,898 articles (23,481 Fake + 21,417 Real)
- **After preprocessing:** 44,034 articles
- **Labels:** 0 = Fake, 1 = Real

> The dataset files (Fake.csv and True.csv) are NOT included in this repository due to size. Download them directly from Kaggle using the link above.

---

## Pipeline
1. **EDA** — Class distribution, text length analysis, subject category exploration
2. **Preprocessing** — Deduplication, lowercasing, truncation to 500 words
3. **Baseline Models** — TF-IDF + Logistic Regression and Linear SVM
4. **DistilBERT Fine-tuning** — 3 epochs, AdamW optimizer, linear warmup scheduler
5. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrices
6. **Explainable AI** — LIME and SHAP word-level explanations

---

## Repository Structure
fake-news-detection/
│
├── README.md                          ← You are here
├── Fake_News_Detection_GradientDescenters.ipynb   ← Main notebook

---

## How to Run

### Step 1 — Open in Google Colab
Upload the notebook to [Google Colab](https://colab.research.google.com/)

### Step 2 — Set Runtime to GPU
Runtime → Change runtime type → **T4 GPU**  
*(Required for DistilBERT training)*

### Step 3 — Download the Dataset
Go to [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and download `Fake.csv` and `True.csv`

### Step 4 — Upload Dataset Files
In Colab, click the folder icon on the left sidebar → upload both `Fake.csv` and `True.csv`

### Step 5 — Run All Cells
Run cells **top to bottom in order** — do not skip any cell  
**Estimated total runtime: ~30-35 minutes on T4 GPU**

---

## Dependencies
All dependencies are installed in Cell 1 of the notebook:
transformers
datasets
lime
shap
torch
scikit-learn
pandas
numpy
matplotlib
seaborn
---

## Hyperparameters
| Parameter | Value |
|---|---|
| Model | distilbert-base-uncased |
| Max token length | 256 |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% of steps) |
| Random seed | 42 |

---

## Explainable AI
We apply two XAI methods to explain individual DistilBERT predictions:
- **LIME** — Identifies words that push predictions toward Fake or Real
- **SHAP** — Token-level Shapley value attribution scores

**Key finding:** The model learned that Reuters wire service datelines strongly indicate Real news, while sensationalist political language indicates Fake news.

---

## Known Limitations
- Dataset covers only 2015-2017 US political news — may not generalize to other domains
- Model partially relies on source signals (Reuters) rather than purely content-level patterns
- Evaluation conducted on a single dataset — cross-dataset testing (LIAR, FakeNewsNet) is future work
- LIME explanations use 300 perturbation samples — increasing to 500+ improves stability

---

## References
Key references from our literature review:
- Devlin et al. (2019) — BERT
- Sanh et al. (2019) — DistilBERT
- Ribeiro et al. (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- Kaliyar et al. (2021) — FakeBERT

Full reference list available in the IEEE report.

