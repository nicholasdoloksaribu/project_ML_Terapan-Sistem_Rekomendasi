# Laporan Proyek Machine Learning – Nicholas Doloksaribu

## Project Overview

Sistem rekomendasi adalah komponen penting dalam layanan digital, terutama pada platform streaming seperti Netflix. Dengan ribuan pilihan film, pengguna kerap kebingungan memilih tontonan. Oleh karena itu, dibutuhkan sistem cerdas untuk memberikan saran yang relevan.

Proyek ini membangun sistem rekomendasi film menggunakan dataset MovieLens. Dua pendekatan utama yang digunakan:
- **Content-Based Filtering**
- **Collaborative Filtering (Neural Network)**

> Referensi:
> - [1] Gomez-Uribe, C.A., & Hunt, N. (2015). *The Netflix Recommender System: Algorithms, Business Value, and Innovation*. ACM Transactions on MIS.
> - [2] Chui, M., et al. (2013). *Disruptive Technologies*. McKinsey Global Institute.

---

##  Business Understanding

### Problem Statements
- Bagaimana membangun sistem rekomendasi berdasarkan konten film?
- Bagaimana memodelkan preferensi pengguna dari histori rating?
- Bagaimana mengevaluasi dan membandingkan kedua pendekatan tersebut?

###  Goals
- Mengembangkan sistem rekomendasi berbasis konten.
- Membangun collaborative filtering menggunakan deep learning.
- Mengevaluasi performa kedua pendekatan dengan metrik yang sesuai.

###  Solution Approach
- **Solusi 1:** TF-IDF + Cosine Similarity untuk content-based filtering.
- **Solusi 2:** Neural Network Collaborative Filtering dengan embedding layer.

---

##  Data Understanding

Dataset: [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/)

| File         | Deskripsi                                    |
|--------------|----------------------------------------------|
| `movies.csv` | Judul dan genre film                         |
| `ratings.csv`| Rating pengguna terhadap film                |
| `tags.csv`   | Tag deskriptif untuk masing-masing film      |

###  Variabel Penting
- `userId`, `movieId`, `rating`, `title`, `genres`, `tag`

###  Exploratory Data Analysis
- Genre populer: Drama, Comedy
- Rentang rating umum: 3–4
- Adanya user/movie dengan rating yang ekstrem

---

##  Data Preparation

- Gabungkan `tags` + `genres` per film
- TF-IDF Vectorizer → Content Feature
- Encode `userId` dan `movieId`
- Normalisasi rating (0–1)
- Train-test split (80:20)

---

##  Modeling

### 1️⃣ Content-Based Filtering
- TF-IDF + Cosine Similarity antar film
- Output: Top-N film paling mirip

### 2️⃣ Collaborative Filtering (Deep Learning)
- Embedding Layer untuk user dan movie
- Dot Product → Prediksi rating
- Optimizer: Adam, Loss: Binary Crossentropy
- Epoch: 10

### Perbandingan

| Pendekatan              | Kelebihan                             | Kekurangan                            |
|-------------------------|----------------------------------------|----------------------------------------|
| Content-Based Filtering | Mudah diinterpretasi, cocok untuk cold-start | Rekomendasi terbatas, tidak personal   |
| Collaborative Filtering | Akurat & personalisasi tinggi         | Tidak explainable, butuh data besar    |

---

##  Evaluation

###  Metrik

- **RMSE (Root Mean Squared Error)**  
  \[
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}
  \]

- **Top-N Recommendation** (CBF):  
  Validasi manual berdasarkan relevansi output.

###  Hasil

- **CBF:** Rekomendasi mirip dan masuk akal.
- **Collaborative Filtering:** RMSE ~0.68 → Akurat dan stabil.

---

##  Penutup

Proyek ini menunjukkan bagaimana dua pendekatan sistem rekomendasi dapat diterapkan untuk meningkatkan pengalaman pengguna dalam memilih film.

###  Pengembangan Selanjutnya
- Tambahkan interface Streamlit
- Integrasi model ke backend dengan FastAPI
- Uji coba dengan data pengguna real-time

---

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, TensorFlow/Keras
- Matplotlib, Seaborn
