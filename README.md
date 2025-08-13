# 🎬 IMDb Movie Review Sentiment Analysis using Custom AvgWord2Vec Embeddings

This project performs **sentiment classification** on the **IMDb 50K Movie Review Dataset** by training a **custom Word2Vec model from scratch** using `gensim`.  
The model represents each review as an **average Word2Vec vector (AvgWord2Vec)**, and these embeddings are then used to train and compare **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** classifiers.

---

## 📌 Project Overview

The goal of this project is to:
1. Preprocess movie review text (tokenization, stopword removal, stemming, etc.).
2. Train a Word2Vec model from scratch on the dataset.
3. Represent each review as the average of its word embeddings (**AvgWord2Vec**).
4. Train and evaluate multiple ML models to classify reviews as **positive** or **negative**.

---

## Project Pipeline

<img width="243" height="552" alt="Image" src="https://github.com/user-attachments/assets/315c9d2d-7fde-46dd-9830-03b6c51570a2" />

## 🛠️ Technologies & Libraries Used

- **Python 3**
- **Libraries**:
  - `gensim` → For training custom Word2Vec embeddings
  - `scikit-learn` → For ML models and evaluation
  - `nltk` → For tokenization, stopword removal, and stemming
  - `numpy` & `pandas` → Data handling
- **Dataset**: IMDb 50K Movie Review Dataset ([Kaggle Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))

---

## 📂 Dataset

- **Size**: 50,000 reviews
- **Classes**:  
  - `positive` sentiment  
  - `negative` sentiment
- **Distribution**: Balanced (25K positive, 25K negative)

---

## ⚙️ Preprocessing Steps

1. **Lowercasing** all text.
2. **Removing HTML tags**.
3. **Removing punctuation**.
4. **Tokenizing** the reviews into words.
5. **Removing stopwords** (using `nltk` stopwords list).
6. **Stemming** words using the Porter Stemmer.
7. **Training custom Word2Vec** model using `gensim` on the processed corpus.
8. Creating **AvgWord2Vec vectors** by averaging all word embeddings in a review.

---

## 📊 Model Training

Three classifiers were trained using the AvgWord2Vec feature vectors:

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | **85%**  |
| Support Vector Machine (SVM) | **85%**  |
| Random Forest        | **82%**  |

---

## 🖥️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/imdb-sentiment-word2vec.git
cd imdb-sentiment-word2vec
