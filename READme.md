


Dimensionality Reduction for Sentiment Analysis using RoBERTa
📌 Overview
This project applies RoBERTa (A Robustly Optimized BERT Pretraining Approach) for sentiment analysis. The model processes tokenized sequences of text to classify sentiment, leveraging advanced contextual embeddings for improved accuracy.

Our pipeline covers:

Feature formatting for RoBERTa

Data preprocessing (tokenization, padding, attention masks)

Model architecture & training strategy

Performance evaluation with detailed misclassification analysis

Domain adaptation for hotel reviews

Exploration of computational trade-offs between contextual models and simpler feature extraction techniques

📂 Feature Format
We use tokenized sequences as input features, generated via the RoBERTa tokenizer.

Tokenization: Splits text into subword units, handling punctuation, casing, and special characters.

Padding: Ensures uniform sequence length (512 tokens).

Attention Masks: Distinguishes real tokens from padding.

Embedding Conversion: Converts tokens and masks into dense vector representations.

🛠 Data Preprocessing Steps
Tokenization – Using RoBERTa tokenizer.

Padding/Truncation – Fixed length: 512 tokens.

Attention Mask Generation – Marks actual tokens vs. padding.

Embedding Conversion – Prepared for model input.

🤖 Model
We employ RoBERTa-base from Hugging Face.

Key Differences from BERT
No Next Sentence Prediction (NSP) – Uses only Masked Language Modeling (MLM).

Dynamic Masking – Masks different tokens at each epoch.

Improved Pretraining – Larger datasets and optimized hyperparameters.

🏗 Model Architecture
Input Layers:

Input_ID → Shape: (1623, 512)

Attention_Mask → Shape: (1623, 512)

RoBERTa Encoder: Generates contextual embeddings.

Dropout Layer: Rate = 0.1 to prevent overfitting.

Output Layer: Single neuron with sigmoid activation for binary classification.

📏 Input & Output Dimensions
Component	Shape
Input IDs	(1623, 512)
Attention Masks	(1623, 512)
Output	(1623, 1)

⚙ Training Strategy
Optimizer: Adam

Learning Rate: 1e-5 (better than 1e-4 which caused underfitting)

Batch Size: 8 (avoids resource exhaustion)

Epochs: 4 (to prevent overfitting)

Loss Function: Binary Cross Entropy (BCE)

Class Balancing: Undersampling to handle imbalance (2164 samples per class after balancing)

📊 Results Analysis
Strengths: Strong contextual understanding, good at detecting sentiment polarity.

Weaknesses: Struggles with sarcasm, irony, mixed sentiments, and nuanced comparisons.

Examples:

Misclassification of positive reviews due to minor negative mentions.

Difficulty in weighing overall sentiment when both pros and cons are present.

🔄 Domain Adaptation for Hotel Reviews
Two approaches:

Transfer Learning: Test product-review-trained model on hotel reviews (binary classification) with tools like LIME for explainability.

Multi-class Classification: Adapt model for 1–10 sentiment scale using softmax activation and categorical cross-entropy loss.

🧠 Improving Robustness
Ensemble Methods: Combine multiple RoBERTa models.

Data Augmentation: Paraphrasing, synonym replacement, back translation.

Bootstrapping: Iterative training on weakly labeled + predicted data.

Active Learning: Human labeling of most informative samples.

⚖ Computational Trade-offs
RoBERTa: High accuracy, contextual awareness, higher resource cost.

TF-IDF / Bag of Words: More efficient but lacks context & semantic relationships.

Model Architechture
![alt text](image.png)