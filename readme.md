# Laporan Proyek Machine Learning – Nicholas Doloksaribu

## Project Overview

Sistem rekomendasi film telah menjadi komponen penting dalam layanan streaming dan platform hiburan modern. Dengan jutaan film yang tersedia, pengguna sering kali kesulitan menemukan konten yang sesuai dengan preferensi mereka. Hal ini dapat menyebabkan pengalaman pengguna yang buruk, penurunan retensi pengguna, dan penurunan pendapatan bagi platform.

Netflix telah menemukan bahwa lebih dari 80% film yang ditonton di platform mereka berasal dari rekomendasi sistem (Gomez-Uribe & Hunt, 2015). Sementara itu, sebuah laporan dari McKinsey menyatakan bahwa sistem rekomendasi yang efektif dapat meningkatkan penjualan hingga 35% (Bounsaythip & Rinta-Runsala, 2001). Hal ini menunjukkan bahwa pengembangan sistem rekomendasi yang akurat dan personal memiliki peran penting dalam meningkatkan keterlibatan pengguna serta keuntungan bisnis.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film yang dapat membantu pengguna menemukan film yang sesuai dengan minat mereka. Dengan menggunakan dataset MovieLens yang berisi rating dan tag film dari ribuan pengguna, proyek ini akan mengimplementasikan dan membandingkan dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering berbasis Deep Learning.

> Referensi:
> - Gomez-Uribe, C.A., & Hunt, N. (2015). The Netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems, 6(4), 1-19. Available at https://dl.acm.org/doi/10.1145/2843948.
> - Bounsaythip, K., & Rinta-Runsala, E. (2001). Overview of data mining for customer behavior modeling. VTT Information Technology, Research Report TTE1-2001-18. Available at https://cris.vtt.fi/en/publications/overview-of-data-mining-for-customer-behavior-modeling-louhi-vers.

---

##  Business Understanding

### Problem Statements
- Bagaimana merancang sistem rekomendasi yang mampu menyarankan film berdasarkan kesamaan karakteristik atau atribut film (menggunakan pendekatan content-based filtering)?
- Bagaimana mengembangkan sistem rekomendasi yang dapat memahami dan menyesuaikan diri dengan preferensi pengguna melalui analisis pola interaksi mereka terhadap film (dengan pendekatan collaborative filtering)?
- Bagaimana mengukur keberhasilan sistem rekomendasi yang telah dikembangkan?

###  Goals
- Merancang sistem rekomendasi berbasis konten yang mampu menyarankan film kepada pengguna dengan mempertimbangkan kesamaan dalam genre dan tag dari film yang pernah disukai.
- Mengimplementasikan pendekatan collaborative filtering dengan memanfaatkan teknik deep learning untuk menangkap pola penilaian (rating) pengguna dan memberikan rekomendasi yang relevan.
- Menilai kinerja model rekomendasi menggunakan metrik evaluasi seperti Root Mean Square Error (RMSE) guna memastikan rekomendasi yang dihasilkan akurat dan sesuai preferensi pengguna.

###  Solution Statements
Untuk mewujudkan tujuan proyek, solusi yang akan diterapkan meliputi dua pendekatan utama:
1. Rekomendasi Berbasis Konten (Content-Based Filtering)
- Menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstrak informasi penting dari genre dan tag film sebagai representasi fitur.
- Menerapkan algoritma Nearest Neighbors dengan pengukuran cosine similarity guna mengidentifikasi film-film yang memiliki karakteristik konten yang mirip.
- Menyusun rekomendasi film dengan mempertimbangkan kesamaan konten terhadap film yang telah disukai pengguna sebelumnya.
2. Rekomendasi Kolaboratif Berbasis Deep Learning (Collaborative Filtering)
- Membangun arsitektur Neural Network yang memanfaatkan embedding layer untuk menangkap representasi laten dari pengguna dan item (film).
- Melatih model pada data rating pengguna untuk mempelajari preferensi dan menghasilkan prediksi terhadap film yang belum ditonton.
- Menyajikan daftar rekomendasi berdasarkan nilai prediksi rating tertinggi dari hasil model.

---

##  Data Understanding
Proyek ini menggunakan dataset MovieLens Small yang dapat diunduh dari GroupLens Research. Dataset ini adalah kumpulan data rating film yang dikembangkan oleh GroupLens Research di University of Minnesota sebagai sumber data untuk penelitian sistem rekomendasi.

**Jumlah Data dan Struktur Dataset**
Dataset terdiri dari beberapa file, namun dalam proyek ini hanya digunakan tiga file utama:
 1. movies.csv
    - Jumlah baris: 9.742 film
    - Jumlah kolom: 3 (movieId, title, genres)
 2. ratings.csv
    - Jumlah baris: 100.836 rating
    - Jumlah kolom: 4 (userId, movieId, rating, timestamp)
 3. tags.csv
    - Jumlah baris: 3.683 tag
    - Jumlah kolom: 4 (userId, movieId, tag, timestamp)

###  Variabel - variabel pada dataset
### movies.csv
- **movieId**: Identifier unik untuk setiap film.
- **title**: Judul film beserta tahun rilis dalam tanda kurung.
- **genres**: Kategori genre film yang dipisahkan dengan pipe (`|`), contoh: `"Comedy|Romance|Drama"`.

### ratings.csv
- **userId**: Identifier unik untuk setiap pengguna.
- **movieId**: Identifier unik untuk setiap film.
- **rating**: Rating yang diberikan pengguna, dengan skala 0.5 hingga 5.0 dalam interval 0.5.
- **timestamp**: Waktu pemberian rating dalam format Unix timestamp.

### tags.csv
- **userId**: Identifier unik untuk setiap pengguna.
- **movieId**: Identifier unik untuk setiap film.
- **tag**: Tag teks yang diberikan pengguna untuk film.
- **timestamp**: Waktu pemberian tag dalam format Unix timestamp.

---

## Kondisi Data

### Missing Values
- Tidak ditemukan missing values pada ketiga dataset (`movies.csv`, `ratings.csv`, dan `tags.csv`).

### Data Duplikat
- Tidak terdapat data duplikat di ketiga dataset tersebut.

### Outlier
- Pada dataset `ratings.csv`, terdapat 4.181 rating yang teridentifikasi sebagai outlier menggunakan metode IQR (Interquartile Range).
- Outlier tersebut tetap dipertahankan dalam analisis karena masih berada dalam rentang nilai rating yang valid (0.5–5.0) dan mencerminkan preferensi unik dari pengguna.


##  Data Preparation
Beberapa teknik data preparation yang diterapkan dalam proyek ini adalah:

### 1. Penggabungan Dataset (Content-Based Filtering)

```python
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = pd.merge(movies, tags_agg, on='movieId', how='left')
```

Tags yang terkait dengan film yang sama digabungkan menjadi satu string teks. Kemudian, data film digabungkan dengan data tag yang telah diagregasi. Hal ini dilakukan untuk memperkaya informasi konten film yang akan digunakan dalam model content-based filtering.

### 2. Pembuatan Fitur Gabungan

```python
movies_with_tags['combined'] = movies_with_tags['genres'].fillna('') + ' ' + movies_with_tags['tag'].fillna('')
```

Genre dan tag film digabungkan menjadi satu fitur teks untuk memudahkan ekstraksi fitur. Nilai null diisi dengan string kosong untuk menghindari error selama proses penggabungan.

### 3. Ekstraksi Fitur Teks dengan TF-IDF

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['combined'])
```

TF-IDF digunakan untuk mengubah teks (genre dan tag) menjadi representasi numerik. TF-IDF efektif dalam menangkap relevansi kata dalam dokumen dengan mempertimbangkan frekuensi kemunculan kata dalam dokumen dan inverse document frequency (IDF) yang memberikan bobot lebih rendah pada kata-kata umum.

### 4. Encoding dan Mapping ID (Collaborative Filtering)

```python
user_ids = df['userId'].unique().tolist()
movie_ids = df['movieId'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

user_encoded_to_user = {i: x for x, i in user_to_user_encoded.items()}
movie_encoded_to_movie = {i: x for x, i in movie_to_movie_encoded.items()}

df['user'] = df['userId'].map(user_to_user_encoded)
df['movie'] = df['movieId'].map(movie_to_movie_encoded)
```

ID pengguna dan film diubah menjadi indeks berurutan (0, 1, 2, ...) untuk memudahkan penggunaan dalam model deep learning. Mapping dari encoded ID ke ID asli juga dibuat untuk memudahkan interpretasi hasil rekomendasi.

### 5. Normalisasi Rating

```python
min_rating = df['rating'].min()
max_rating = df['rating'].max()

df['rating_normalized'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
```

Rating dinormalisasi ke rentang 0-1 untuk mempercepat konvergensi model dan meningkatkan stabilitas selama pelatihan. Normalisasi menggunakan min-max scaling yang memetakan nilai minimum ke 0 dan nilai maksimum ke 1.

### 6. Pembagian Data untuk Training dan Validasi

```python
x = df[['user', 'movie']].values
y = df['rating_normalized'].values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
```

Data dibagi menjadi set training (80%) dan validasi (20%) untuk melatih dan mengevaluasi model. Random state ditetapkan untuk memastikan reprodusibilitas hasil.

**Contoh Hasil Rekomendasi:**

Untuk film "Pinocchio (1940)", sistem merekomendasikan film-film animasi serupa yang memiliki genre dan tag yang mirip.

**Kelebihan:**
- Tidak memerlukan data dari pengguna lain (cold start)
- Dapat memberikan rekomendasi untuk film baru yang belum dinilai
- Mampu menjelaskan alasan di balik rekomendasi (explainability)

**Kekurangan:**
- Terbatas pada fitur yang diekstrak (genre dan tag)
- Tidak mempertimbangkan kualitas film atau preferensi spesifik pengguna
- Cenderung merekomendasikan film yang sangat mirip dan kurang beragam

## Modeling and Result

Dalam proyek ini, dua pendekatan sistem rekomendasi diimplementasikan: Content-Based Filtering dan Collaborative Filtering dengan Deep Learning.

### 1. Content-Based Filtering

Model content-based filtering diimplementasikan menggunakan algoritma Nearest Neighbors dengan metrik cosine similarity:

```python
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

def recommend_content(title, top_n=5):
    idx = title_to_index[title]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
    indices = indices.flatten()[1:]
    return movies_with_tags.iloc[indices][['title', 'genres']]
```

**Cara Kerja:**
1. TF-IDF digunakan untuk mengekstrak fitur dari teks gabungan genre dan tag film
2. Algoritma k-nearest neighbors mencari film dengan representasi TF-IDF yang paling mirip
3. Cosine similarity digunakan sebagai metrik untuk mengukur kemiripan antar film
4. Film terdekat (kecuali film itu sendiri) direkomendasikan kepada pengguna

**Parameter yang Digunakan:**
- **TfidfVectorizer**:
  - `stop_words='english'`: Menghilangkan kata-kata umum dalam bahasa Inggris seperti "the", "a", "in", dll.
  - Parameter lain menggunakan nilai default

- **NearestNeighbors**:
  - `metric='cosine'`: Menggunakan cosine similarity sebagai metrik jarak
  - `algorithm='brute'`: Menggunakan brute force search untuk mencari tetangga terdekat
  - `n_neighbors=top_n+1`: Mencari top_n+1 tetangga terdekat (termasuk film itu sendiri)


**Contoh Hasil Rekomendasi:**

Untuk film "Pinocchio (1940)", sistem merekomendasikan film-film animasi serupa yang memiliki genre dan tag yang mirip.

**Kelebihan:**
- Tidak memerlukan data dari pengguna lain (cold start)
- Dapat memberikan rekomendasi untuk film baru yang belum dinilai
- Mampu menjelaskan alasan di balik rekomendasi (explainability)

**Kekurangan:**
- Terbatas pada fitur yang diekstrak (genre dan tag)
- Tidak mempertimbangkan kualitas film atau preferensi spesifik pengguna
- Cenderung merekomendasikan film yang sangat mirip dan kurang beragam



### 2. Collaborative Filtering dengan Deep Learning

Model collaborative filtering diimplementasikan menggunakan Neural Network dengan lapisan embedding:

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size, 
                                               embeddings_initializer='he_normal', 
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, 
                                                embeddings_initializer='he_normal', 
                                                embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)
```

**Parameter yang Digunakan:**
- **Model Arsitektur**:
  - `embedding_size=50`: Dimensi embedding untuk representasi pengguna dan film
  - `embeddings_initializer='he_normal'`: Inisialisasi bobot menggunakan distribusi He Normal
  - `embeddings_regularizer=keras.regularizers.l2(1e-6)`: L2 regularization untuk mencegah overfitting

- **Kompilasi Model**:
  - `loss=tf.keras.losses.BinaryCrossentropy()`: Fungsi loss untuk nilai target yang dinormalisasi (0-1)
  - `optimizer=keras.optimizers.Adam(learning_rate=0.001)`: Optimizer Adam dengan learning rate 0.001
  - `metrics=[tf.keras.metrics.RootMeanSquaredError()]`: Metrik evaluasi menggunakan RMSE

- **Pelatihan Model**:
  - `batch_size=64`: Jumlah sampel yang diproses dalam satu iterasi
  - `epochs=10`: Jumlah iterasi pelatihan pada seluruh dataset
  - `validation_data=(x_val, y_val)`: Data validasi untuk evaluasi performa model

```python
model = RecommenderNet(num_users, num_movies, embedding_size=50)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_val, y_val)
)
```

**Cara Kerja:**
1. Setiap pengguna dan film diubah menjadi representasi vektor berdimensi 50 dalam ruang embedding.
2. Model belajar menangkap pola laten dari data rating yang diberikan pengguna terhadap film.
3. Prediksi rating dibuat dengan menghitung hasil perkalian titik (dot product) antara vektor pengguna dan film.
4. Penyesuaian tambahan dilakukan dengan menambahkan bias khusus pengguna dan film untuk mencerminkan preferensi umum.
5. Hasil prediksi kemudian dilewatkan ke fungsi sigmoid agar berada dalam rentang nilai 0 sampai 1.
6. Proses pelatihan model berfokus pada meminimalkan perbedaan antara rating prediksi dan rating asli menggunakan fungsi loss binary cross-entropy.

**Contoh Hasil Rekomendasi:**

Untuk pengguna dengan ID 356, sistem menampilkan:
1. Film dengan rating tertinggi yang telah diberikan oleh pengguna
2. Top 10 rekomendasi film berdasarkan prediksi model

**📌 Film dengan Rating Tertinggi dari Pengguna**
| Judul Film                                                         | Genre                                |
|-------------------------------------------------------------------|--------------------------------------|
| So I Married an Axe Murderer (1993)                               | Comedy &#124; Romance &#124; Thriller |
| Wallace & Gromit: The Best of Aardman Animation (1996)            | Adventure &#124; Animation &#124; Comedy |
| Superbad (2007)                                                   | Comedy                               |
| Lars and the Real Girl (2007)                                     | Comedy &#124; Drama                   |
| Battlestar Galactica: Razor (2007)                                | Action &#124; Drama &#124; Sci-Fi &#124; Thriller |


**🔮 Top-10 Film Rekomendasi untuk User**

| No | Judul Film                                                 | Genre                                |
|----|------------------------------------------------------------|--------------------------------------|
| 1  | Star Wars: Episode IV - A New Hope (1977)                 | Action &#124; Adventure &#124; Sci-Fi |
| 2  | Pulp Fiction (1994)                                       | Comedy &#124; Crime &#124; Drama &#124; Thriller |
| 3  | Schindler's List (1993)                                   | Drama &#124; War                     |
| 4  | Godfather, The (1972)                                     | Crime &#124; Drama                   |
| 5  | Rear Window (1954)                                        | Mystery &#124; Thriller              |
| 6  | Star Wars: Episode V - The Empire Strikes Back (1980)    | Action &#124; Adventure &#124; Sci-Fi |
| 7  | Goodfellas (1990)                                         | Crime &#124; Drama                   |
| 8  | Boondock Saints, The (2000)                               | Action &#124; Crime &#124; Drama &#124; Thriller |
| 9  | In America (2002)                                         | Drama &#124; Romance                 |
| 10 | Prestige, The (2006)                                      | Drama &#124; Mystery &#124; Sci-Fi &#124; Thriller |


**Kelebihan:**
- Mampu mengidentifikasi pola preferensi pengguna yang kompleks dan tidak langsung terlihat.
- Dapat memberikan rekomendasi film yang beragam, termasuk yang tidak secara eksplisit terkait dengan riwayat tontonan pengguna.
- Menyajikan rekomendasi yang lebih personal dan variatif sesuai dengan kebiasaan dan preferensi unik pengguna.

**Kekurangan:**
- Membutuhkan jumlah data interaksi pengguna yang cukup agar model dapat bekerja efektif (masalah cold start pada pengguna baru).
- Memerlukan sumber daya komputasi yang lebih besar dibandingkan metode berbasis konten.
- Tingkat keterjelasan alasan rekomendasi (explainability) lebih rendah dibandingkan pendekatan content-based filtering.

## Evaluation

Evaluasi model dalam proyek ini dilakukan dengan beberapa metrik dan juga dikaitkan dengan business understanding yang telah ditetapkan sebelumnya.

### 1. Root Mean Squared Error (RMSE)

RMSE digunakan untuk mengevaluasi model collaborative filtering berbasis deep learning. RMSE mengukur akar kuadrat dari rata-rata selisih kuadrat antara rating prediksi dan rating sebenarnya.

Formula RMSE:

$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

Dimana:
- $y_i$ adalah rating sebenarnya
- $\hat{y}_i$ adalah rating prediksi
- $n$ adalah jumlah sampel

Hasil evaluasi menunjukkan:
- RMSE pada data latih: 0.196
- RMSE pada data validasi: 0.203

Nilai RMSE yang rendah (mendekati 0) menunjukkan performa model yang baik. Nilai RMSE yang mirip antara data latih dan validasi juga menunjukkan bahwa model tidak mengalami overfitting.

### 2. Relevance dan Diversity

Untuk model content-based filtering, evaluasi dilakukan secara kualitatif dengan melihat relevansi rekomendasi:

- **Relevance**: Rekomendasi yang diberikan untuk film "Pinocchio (1940)" menunjukkan film-film animasi yang serupa, menandakan bahwa model berhasil menangkap kesamaan konten.

- **Diversity**: Meskipun rekomendasi content-based filtering cenderung kurang beragam, namun masih mampu merekomendasikan film dengan variasi yang cukup dalam subgenre yang sama.

### 3. User-Centered Evaluation

Evaluasi berpusat pada pengguna dilakukan dengan melihat kesesuaian rekomendasi dengan preferensi pengguna:

- **Personalisasi**: Untuk pengguna dengan ID 356, model collaborative filtering memberikan rekomendasi film yang sesuai dengan preferensi pengguna berdasarkan pola rating mereka.

- **Novelty**: Model collaborative filtering mampu merekomendasikan film yang mungkin belum pernah dilihat pengguna tetapi memiliki kemungkinan disukai berdasarkan preferensi mereka.

### Keterkaitan dengan Business Understanding

#### Pernyataan Masalah 1: Sistem Rekomendasi berbasis Konten
- **Goal**: Merancang sistem rekomendasi yang dapat menyarankan film dengan genre dan tag serupa.
- **Solusi**: Implementasi model content-based filtering menggunakan TF-IDF dan Nearest Neighbors.
- **Hasil**: Model berhasil memberikan reko
mendasi film yang memiliki kesamaan konten (genre dan tag) dengan film referensi. Misalnya, untuk film "Pinocchio (1940)", sistem merekomendasikan film-film animasi serupa.
- **Dampak Bisnis**: Membantu mengatasi tantangan cold start pada film baru, meningkatkan visibilitas konten katalog, dan mempercepat pengguna menemukan film yang sejenis dengan preferensi mereka.

#### Pernyataan Masalah 2: Sistem Rekomendasi berbasis Interaksi Pengguna
- **Goal**: Mengembangkan sistem yang dapat mempelajari dan memodelkan kebiasaan pengguna melalui data interaksi.
- **Solusi**: Implementasi model collaborative filtering dengan deep learning menggunakan lapisan embedding.
- **Hasil**: Model mencapai RMSE 0.203 pada data validasi, menunjukkan kemampuan yang baik dalam memprediksi rating. Model berhasil memberikan rekomendasi personal untuk pengguna berdasarkan pola rating mereka.
- **Dampak Bisnis**: Menyediakan pengalaman pengguna yang lebih personal dan relevan, mendorong keterlibatan lebih tinggi, serta dapat meningkatkan tingkat konversi pengguna terhadap konten hingga 35%.

#### Pernyataan Masalah 3: Evaluasi Sistem Rekomendasi
- **Goal**: Menentukan seberapa baik sistem rekomendasi berfungsi dalam menyarankan film yang sesuai.
- **Solusi**: Menggunakan RMSE untuk model collaborative filtering dan evaluasi kualitatif untuk content-based filtering.
- **Hasil**: RMSE sebesar 0.203 menunjukkan akurasi prediksi yang baik. Evaluasi kualitatif menunjukkan bahwa rekomendasi content-based filtering relevan dengan film referensi.
- **Dampak Bisnis**: Evaluasi yang terukur memudahkan proses pemantauan performa sistem dan memberikan dasar untuk pengembangan lanjutan, yang berkontribusi pada peningkatan loyalitas dan kepuasan pengguna.

### Dampak Solusi Statement

1. **Content-Based Filtering dengan TF-IDF dan Nearest Neighbors**:
   - **Dampak**: Solusi ini berhasil memberikan rekomendasi berdasarkan konten film, membantu mengatasi cold-start problem, dan meningkatkan eksposur katalog film yang mungkin tidak populer tetapi relevan dengan minat pengguna.
   - **Bisnis Impact**: Model ini berkontribusi pada perluasan jangkauan konten yang terekspos kepada pengguna, memperkuat fitur eksplorasi konten baru (content discovery), dan membantu mempertahankan pengguna dengan menawarkan alternatif tontonan yang sesuai setelah mereka menyelesaikan film favoritnya. Dengan demikian, potensi penurunan churn rate dapat ditekan.

2. **Collaborative Filtering dengan Deep Learning**:
   - **Dampak**: Solusi ini berhasil mempelajari pola tersembunyi dalam preferensi pengguna dan memberikan rekomendasi personal yang relevan, meningkatkan pengalaman pengguna.
   - **Bisnis Impact**: Pendekatan ini mendorong peningkatan keterlibatan (engagement) dengan menghadirkan rekomendasi yang semakin relevan dan spesifik. Hasilnya, waktu interaksi pengguna di dalam platform dapat meningkat, loyalitas pengguna cenderung lebih kuat, dan dalam konteks layanan berbayar, sistem ini memiliki potensi untuk mendorong tingkat konversi pelanggan.

### Perbandingan dan Kesimpulan

Kedua model rekomendasi yang dikembangkan dalam proyek ini memiliki kelebihan dan kekurangan masing-masing:

1. **Content-Based Filtering**  sangat efektif dalam situasi ketika informasi tentang pengguna masih terbatas atau ketika ingin merekomendasikan film-film baru yang belum memiliki banyak interaksi. Sistem ini memberikan hasil rekomendasi yang dapat dijelaskan dengan jelas karena berdasarkan kesamaan atribut konten seperti genre dan tag.

2. **Collaborative Filtering dengan Deep Learning** unggul dalam menyajikan rekomendasi yang lebih dipersonalisasi. Dengan mempelajari pola rating pengguna, sistem ini mampu menangkap preferensi yang lebih kompleks dan menyarankan film yang mungkin tidak secara eksplisit mirip, namun disukai oleh pengguna dengan pola yang serupa. Evaluasi menggunakan RMSE menghasilkan nilai 0.203, menunjukkan kinerja prediktif yang baik.

Dari sisi implementasi bisnis, pendekatan yang menggabungkan kedua metode ini menjadi sistem hybrid recommendation dinilai paling optimal. Content-based filtering membantu dalam mengatasi tantangan cold-start dan mendukung eksplorasi film yang belum populer, sementara collaborative filtering memperkuat personalisasi dan meningkatkan keterlibatan pengguna.
