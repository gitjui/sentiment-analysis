

# Dimensionality Reduction for Sentiment Analysis with RoBERTa

## 📌 Overview

This project leverages **RoBERTa** (*A Robustly Optimized BERT Pretraining Approach*) for sentiment analysis.
It processes tokenized text sequences to classify sentiment, using **contextual embeddings** for richer semantic understanding and improved accuracy.

Our pipeline includes:

* **Feature formatting** for RoBERTa inputs
* **Data preprocessing** (tokenization, padding, attention masks)
* **Model architecture & training strategy**
* **Performance evaluation** with misclassification insights
* **Domain adaptation** for hotel reviews
* **Computational trade-off analysis** vs. simpler models

---

## 📂 Feature Format

Input features are tokenized sequences produced by the RoBERTa tokenizer:

* **Tokenization** → Splits text into subword units, handling punctuation, casing, and special characters.
* **Padding** → Uniform sequence length of **512 tokens**.
* **Attention Masks** → Flags real tokens vs. padding.
* **Embedding Conversion** → Converts tokens & masks into dense vector representations.

---

## 🛠 Data Preprocessing Steps

1. **Tokenization** – Using Hugging Face RoBERTa tokenizer
2. **Padding/Truncation** – Fixed length = 512 tokens
3. **Attention Mask Generation** – Distinguish real tokens from padding
4. **Embedding Preparation** – Convert to model-ready tensors

---

## 🤖 Model

We use **RoBERTa-base** from Hugging Face.

**Key differences from BERT:**

* **No NSP** → Only Masked Language Modeling (MLM)
* **Dynamic Masking** → New tokens masked every epoch
* **Enhanced Pretraining** → Larger corpus + tuned hyperparameters

---

## 🏗 Model Architecture

**Input Layers:**

* **Input IDs** → Shape: `(1623, 512)`
* **Attention Masks** → Shape: `(1623, 512)`

**Processing Layers:**

* **RoBERTa Encoder** → Generates contextual embeddings
* **Dropout** → `rate=0.1` to reduce overfitting

**Output Layer:**

* Single neuron with **sigmoid activation** (binary classification)

| Component       | Shape       |
| --------------- | ----------- |
| Input IDs       | (1623, 512) |
| Attention Masks | (1623, 512) |
| Output          | (1623, 1)   |

![RoBERTa Model Architecture]
<img width="3600" height="900" alt="image" src="https://github.com/user-attachments/assets/3c7a8a44-eb70-46d7-a75c-a7e175d34735" />


---

## ⚙ Training Strategy

* **Optimizer** → Adam
* **Learning Rate** → `1e-5` (avoids underfitting seen with `1e-4`)
* **Batch Size** → 8 (GPU memory-friendly)
* **Epochs** → 4 (prevents overfitting)
* **Loss Function** → Binary Cross-Entropy (BCE)
* **Class Balancing** → Undersampling to 2,164 samples per class

---

## 📊 Results Analysis

**Strengths:**

* Strong contextual understanding
* Accurately detects sentiment polarity

**Weaknesses:**

* Struggles with sarcasm, irony, mixed sentiments
* Can misclassify when both pros and cons are present

---

## 🔄 Domain Adaptation for Hotel Reviews

**Approaches:**

1. **Transfer Learning** – Apply product-review-trained model directly to hotel reviews; use LIME for explainability.
2. **Multi-Class Classification** – Map 1–10 rating scale using **softmax** and categorical cross-entropy.

---

## 🧠 Improving Robustness

* **Ensemble Methods** – Combine multiple RoBERTa variants
* **Data Augmentation** – Paraphrasing, synonym replacement, back-translation
* **Bootstrapping** – Train iteratively on weakly labeled + predicted data
* **Active Learning** – Label most informative samples first

---

## ⚖ Computational Trade-offs

| Approach       | Accuracy | Contextual Awareness | Compute Cost |
| -------------- | -------- | -------------------- | ------------ |
| **RoBERTa**    | ✅ High   | ✅ Strong             | ❌ High       |
| **TF-IDF/BOW** | ❌ Lower  | ❌ Weak               | ✅ Low        |

