# Laporan Proyek Machine Learning - Zahwa Genoveva
---
## Domain Proyek 
---
Domain proyek yang dipilih dalam proyek _machine learning_ ini adalah mengenai Pertanian dengan judul proyek "Prediksi kualitas buah apel".

* #####  Latar Belakang
  ![](https://cdn1-production-images-kly.akamaized.net/xBRT773xJ9vvZx12J6fN1IIVhko=/1200x675/smart/filters:quality(75):strip_icc():format(jpeg)/kly-media-production/medias/2792644/original/076377400_1556614591-aaron-blanco-tejedor-390113-unsplash.jpg)
Apel (Malus domestica) adalah salah satu jenis buah yang sangat diminati masyarakat berkat variasi rasa yang ditawarkannya. Di industri pertanian dan pemasaran, kualitas buah apel menjadi faktor penting yang sangat memengaruhi nilai jual dan minat konsumen [[1]](https://protan.studentjournal.ub.ac.id/index.php/protan/article/view/1). Untuk itu, berbagai metode telah dikembangkan guna memprediksi dan meningkatkan kualitas buah ini. Meskipun pendekatan konvensional seperti pengukuran manual dan analisis laboratorium masih sering digunakan, metode ini cenderung membutuhkan waktu yang lama serta tidak selalu memberikan hasil yang akurat.

  Dalam laporan ini, Dibangunlah beberapa model machine learning dengan tujuan untuk memprediksi kualitas apel berdasarkan fitur-fitur yang telah diukur. Model-model yang digunakan meliputi K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), Naive Bayes, Decision Tree, dan XGBoost. Setiap algoritma memiliki kelebihan dan kekurangan yang berbeda dalam menangani data, yang kami evaluasi melalui metrik seperti accuracy, precision, dan recall [[3]](https://doi.org/10.47970/siskom-kb.v4i1.169). Penggunaan metode ini diharapkan dapat meningkatkan akurasi prediksi kualitas apel dan membantu produsen dalam pengambilan keputusan yang lebih baik terkait klasifikasi produk untuk konsumsi atau penjualan.

## Business Understanding
---
#### Problem Statements
berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
* Algoritma machine learning mana yang paling efektif untuk menyelesaikan masalah klasifikasi kualitas apel?
* Bagaimana cara mengatasi proses penilaian kualitas apel yang memerlukan waktu lama dan seringkali tidak akurat?
* Bagaimana cara mengurangi biaya dan waktu yang dibutuhkan dalam evaluasi kualitas apel dengan menggunakan metode konvensional seperti pengukuran manual dan analisis laboratorium?
* Bagaimana produsen dapat menggunakan sistem yang lebih efisien dan akurat untuk mengklasifikasikan kualitas apel berdasarkan beberapa karakteristik seperti ukuran, berat, kematangan, keasaman, dan tekstur?
* Bagaimana mengatasi ketidakkonsistenan dalam kualitas apel yang berdampak negatif pada nilai jual dan kepuasan konsumen?       


#### Goals
* Meningkatkan efisiensi proses penilaian kualitas apel dengan menggunakan metode berbasis machine learning.
* Meningkatkan akurasi dalam prediksi kualitas apel untuk memastikan konsistensi produk.
* Mengoptimalkan keputusan bisnis produsen dalam klasifikasi apel untuk penjualan atau konsumsi.
* Memperkuat daya saing produk apel dengan memastikan kualitas yang lebih konsisten dan dapat diandalkan, sehingga meningkatkan kepuasan konsumen dan nilai jual.

#### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
* Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Melakukan _drop_ kolom pada kolom ID.
    * Mengatasi masalah data yang kosong dengan nilai rata-rata kolom (_mean substitution_).
    * Melakukan Encoding terhadap kolom yang bertipe _object_.
    * Mengatasi masalah data tidak seimbang dengan _resample_.
    * Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
    * Melakukan _Standard Scaler_.


* Menggunakan algoritma machine learning seperti K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), Naive Bayes, Decision Tree, dan XGBoost untuk membangun model prediksi kualitas apel berdasarkan fitur-fitur yang diukur (misalnya, ukuran, berat, kematangan, keasaman, dan tekstur).

    
## Data Understanding
---
![Image of Dataset](https://i.postimg.cc/CKJ0sBXT/Screenshot-2024-10-10-204041.png)
Informasi dataset dapat dilihat pada **Tabel 1. Informasi dataset** dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data)
Lisensi | Data files © Original Authors
Kategori | Computer Science, Education, Food, Data Visualization, Classification, Exploratory Data Analysis
Jenis dan Ukuran Berkas | CSV (387.65 kB)
---

Pada tahap ini, kita akan menganalisis struktur dan karakteristik dataset yang digunakan. Dataset terdiri dari 9 kolom dan 4001 baris, dengan rincian sebagai berikut:

### Deskripsi Data
   * Jumlah data: 4001 baris.
   * Jumlah kolom: 9 kolom.
   * Tipe data:
      * 7 kolom numerik (float64)
      * 2 kolom kategorikal (object)
   * Kolom:
      * A_id: Identifikasi unik untuk setiap apel.
      * Size: Ukuran apel.
      * Weight: Berat apel.
      * Sweetness: Tingkat kemanisan apel.
      * Crunchiness: Tingkat kekeruhan apel.
      * Juiciness: Tingkat kejuisan apel.
      * Ripeness: Tingkat kematangan apel.
      * Acidity: Tingkat keasaman apel.
      * Quality: Label kualitas apel (baik/buruk).
   * Kondisi Data:
      * Data tidak sepenuhnya lengkap, terdapat missing value pada kolom Acidity dan Quality.
      * Untuk data numerik, rata-rata, standar deviasi, serta nilai minimum dan maksimum telah dihitung. Tabel statistik deskriptif dari data
        tersebut ditampilkan di bawah.
    * Statistik Deskriptif:
      
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Apple Quality_

**Tabel 2. Tabel Deskripsi**

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| 0.0 | -3.970049 |-2.512336 | 5.346330 |-1.012009 | 1.844900 |0.329840	| -0.491590483  |good |
| 1.0 | -1.195217 |-2.839257 | 3.664059 |1.588232 | 0.853286 | 0.867530 | -0.722809367  |good |
| 2.0 | -0.292024 |	-1.351282 | -1.738429 | -0.342616 | 2.838636 |-0.038033	| 2.621636473  |bad |
| 3.0 | -0.657196 |-2.271627 | 1.324874 |-0.097875 | 3.637970 |-3.413761	| 0.790723217  |good |
| 4.0 | 1.364217 |-1.296612 | -0.384658 | -0.553006 | 3.030874 | -1.303849	| 0.501984036  |good |
  

### Univariate Analysis
 
Selanjutnya, akan dilakukan proses analisis data dengan teknik Univariate EDA. Pada kasus ini semua fiturnya adalah fitur numerik dan tidak ada fitur kategorikal. Sehingga hanya perlu dilakukan analisa terhadap fitur numerik, sebagai berikut:

#### Analisa Fitur Numerik

Untuk melihat distribusi data pada tiap fitur akan digunakan visualisasi dengan histogram sebagai berikut:

![histogram](https://i.postimg.cc/CxqtBcgM/ultra.png)
**Gambar 1. Visualisasi Histogram**

Dari hasil visualisasi histogram di atas, kita bisa memperoleh beberapa informasi, antara lain:

Distribusi fitur quality (target) cenderung miring ke kanan (right-skewed).

Karena beberapa fitur belum terdistribusi normal hal ini akan berimplikasi pada model, maka selanjutnya kita lakukan transformasi data (non-linear scaling). Namun, sebelum itu kita cek terlebih dahulu hubungan antara fitur numerik tersebut.

### Multivariate Analysis

#### Hubungan Antara Fitur Numerik

Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut:

![grafik_pairplot](https://i.postimg.cc/ncbjvz84/multi.png)

**Gambar 2. Visualisasi Hubungan antara Fitur Numerik dengan pairplot()**

Secara keseluruhan, pairplot ini membantu kita memahami kompleksitas hubungan antara berbagai karakteristik (seperti ukuran, berat, kemanisan, dsb.) dari suatu objek. Kita bisa melihat variabel mana yang saling terkait, variabel mana yang tidak, dan variabel mana yang mungkin mempengaruhi kualitas.

#### Korelasi antara Fitur Numerik

Untuk mengevaluasi skor korelasi hubungan antara fitur numerik, akan digunakan fungsi corr() dengan output sebagai berikut.

![matriks](https://i.postimg.cc/25wFtWYN/matriks.png)

**Gambar 3. Korelasi antara Fitur Numerik**

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Berdasarkan matriks korelasi ini, kita bisa menyimpulkan bahwa kualitas suatu objek dipengaruhi oleh beberapa faktor seperti ukuran, juiciness, acidity, sweetness, dan ripeness. Objek yang lebih besar, lebih juicy, lebih asam, dan kurang manis serta kurang matang cenderung memiliki kualitas yang lebih baik.



## Data Preparation
---
Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data:

### 1. Menangani Missing Values

Untuk mendeteksi missing value digunakan fungsi isnull().sum() dan diperoleh: **Tabel 3. Hasil Deteksi Missing Value**
| Variable     | Missing Value | Metode Penanganan |
|--------------|---------------|-------------------|
| Size         | 1             | Imputasi Mean     |
| Weight       | 1             | Imputasi Mean     |
| Sweetness    | 1             | Imputasi Mean     |
| Crunchiness  | 1             | Imputasi Mean     |
| Juiciness    | 1             | Imputasi Mean     |
| Ripeness     | 1             | Imputasi Mean     |
| Acidity      | 0             | Tidak ada         |
| Quality      | 1             | Imputasi Modus    |

Langkah yang diambil:
- Missing values pada kolom numerik akan diisi menggunakan rata-rata (**mean**).
- Missing values pada kolom kategorikal akan diimputasi menggunakan nilai yang paling sering muncul (**modus**).

### 2. Menangani Outliers

Untuk menangani outliers, metode **Interquartile Range (IQR)** digunakan. IQR mengidentifikasi data yang berada di luar rentang normal, yaitu di luar batas [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

**Tabel 4. Visualisasi Boxplot Sebelum dan Sesudah Metode IQR**:

| Boxplot Sebelum IQR        | Boxplot Setelah IQR        |
|----------------------------|----------------------------|
| ![Before](https://i.postimg.cc/BQ72GNWp/sebelum.png)                | ![After](https://i.postimg.cc/X7PBPJFD/sesudah.png)                |


### 3. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outliers

Setelah dilakukan pembersihan outliers dengan metode **IQR**, jumlah data yang tersisa dibandingkan dengan jumlah data awal adalah seperti pada **Tabel 5. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outlier**:

| Jumlah Data Sebelum  | Jumlah Data Setelah |
|----------------------|---------------------|
| 4001          |           3790          |


Bagian ini menunjukkan proses eksplorasi data yang menyeluruh, mulai dari penanganan missing values hingga outliers, yang dilakukan sebelum melanjutkan ke tahap pemodelan machine learning.


### 4. Train Test Split

Dataset akan dibagi menjadi data latih (train) dan data uji (test). Tujuan langkah ini sebelum proses lainnya adalah agar tidak mengotori data uji dengan informasi yang didapat dari data latih. Contoh pada proses standarisasi dimana jika belum di bagi menjadi data latih dan uji, maka keduanya akan terkena transformasi data yang menggunakan informasi (mean dan standard deviation) dari gabungan data latih dan uji. Hal ini berpotensi menimbulkan kebocoran data (data leakage). Oleh karena itu langkah awal sebelum melakukan tranformasi data adalah membagi dataset terlebih dahulu [3].

Pada kasus ini akan menggunakan proporsi pembagian sebesar 90:10 dengan fungsi train_test_split dari sklearn dengan output sebagai berikut.

**Tabel 6. Jumlah Data Latih dan Uji**

Jumlah Total Data | Jumlah Data Latih | Jumlah Data Uji
----------------- | ----------------- | ---------------
  4000 | 3200 | 800

### 5. Standarisasi
Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada kasus ini akan digunakan metode StandarScaler() dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.


### a. Standarisasi (Normalisasi Fitur)

Pada langkah ini, kita akan menstandarisasi fitur dengan menggunakan `StandardScaler`, sehingga setiap fitur memiliki rata-rata 0 dan deviasi standar 1.

**Tabel 7. Hasil Proses Standarisasi Pada Setiap Fitur Pada Data Latih**

| Deskripsi                             | Keterangan                              |
|---------------------------------------|-----------------------------------------|
| Metode Standarisasi                   | StandardScaler                          |
| Rata-rata fitur setelah standarisasi   | 0                                       |
| Deviasi standar fitur setelah standarisasi | 1                                       |

### b. Normalisasi

Selain standarisasi, kita juga menerapkan normalisasi menggunakan `MinMaxScaler`, yang mengubah skala fitur ke dalam rentang [0, 1].

#### Hasil Normalisasi:

**Tabel 8. Hasil Normalisasi**

|     Size     |    Weight    |  Sweetness  |  Crunchiness |  Juiciness   |  Ripeness   |   Acidity   |
|--------------|--------------|-------------|--------------|--------------|-------------|-------------|
|  1.027890    | -1.001988    |  0.084334   |  0.393453    |  0.271185    |  1.108047   | -0.627140   |
|  0.173070    |  0.852012    | -0.323051   | -0.354721    | -0.106767    |  0.265260   | -1.075288   |
| -1.041694    | -1.165875    | -0.192792   | -1.034102    | -0.324381    |  2.470441   | -0.901013   |
| -0.456591    |  0.285175    |  1.948610   | -0.588818    |  1.024831    | -0.509328   | -0.057273   |
| -0.816011    | -0.146890    | -0.177478   |  2.012460    |  0.249208    |  0.139939   | -0.030334   |

### c. Perbandingan Nilai Asli dan Nilai Normalisasi

**Tabel 9. Perbandingan Nilai Asli dan Nilai Normalisasi**

| Original Size | Original Weight | Original Sweetness | Original Crunchiness | Original Juiciness | Original Ripeness | Original Acidity | Normalized Size | Normalized Weight | Normalized Sweetness | Normalized Crunchiness | Normalized Juiciness | Normalized Ripeness | Normalized Acidity |
|---------------|------------------|---------------------|----------------------|--------------------|--------------------|-------------------|------------------|---------------------|----------------------|-----------------------|---------------------|---------------------|---------------------|
|  1.027890     | -1.001988        |  0.084334           |  0.393453            |  0.271185          |  1.108047          | -0.627140         |  1.027890        | -1.001988           |  0.084334            |  0.393453             |  0.271185           |  1.108047           | -0.627140           |
|  0.173070     |  0.852012        | -0.323051           | -0.354721            | -0.106767          |  0.265260          | -1.075288         |  0.173070        |  0.852012           | -0.323051            | -0.354721             | -0.106767           |  0.265260           | -1.075288           |
| -1.041694     | -1.165875        | -0.192792           | -1.034102            | -0.324381          |  2.470441          | -0.901013         | -1.041694        | -1.165875           | -0.192792            | -1.034102            | -0.324381           |  2.470441           | -0.901013           |
| -0.456591     |  0.285175        |  1.948610           | -0.588818            |  1.024831          | -0.509328          | -0.057273         | -0.456591        |  0.285175           |  1.948610            | -0.588818            |  1.024831           | -0.509328           | -0.057273           |
| -0.816011     | -0.146890        | -0.177478           |  2.012460            |  0.249208          |  0.139939          | -0.030334         | -0.816011        | -0.146890           | -0.177478            |  2.012460            |  0.249208           |  0.139939           | -0.030334           |

### d. Data setelah Normalisasi

**Tabel 10. Setelah Normalisasi**

|     Size     |    Weight    |  Sweetness  |  Crunchiness |  Juiciness   |  Ripeness   |   Acidity   |
|--------------|--------------|-------------|--------------|--------------|-------------|-------------|
|  0.63683187  |  0.35305009  |  0.49645027 |  0.55451442  |  0.52664718  |  0.64304288 |  0.40105263 |
|  0.51523605  |  0.58231483  |  0.43646357 |  0.47743345  |  0.47180237  |  0.52263894 |  0.33565786 |
|  0.34243895  |  0.33278396  |  0.45564396 |  0.40744007  |  0.44022423  |  0.83767975 |  0.36108855 |
|  0.42566827  |  0.51222009  |  0.77096161 |  0.45331553  |  0.63600913  |  0.41197840 |  0.48420897 |
|  0.37454175  |  0.45879104  |  0.45789899 |  0.72131318  |  0.52345814  |  0.50473515 |  0.48814003 |



Normalisasi merupakan proses penting dalam tahap persiapan data, yang bertujuan untuk mengubah skala fitur agar berada dalam rentang yang sama. Dalam konteks dataset ini, normalisasi membantu memastikan bahwa setiap atribut, seperti Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, dan Acidity, memiliki kontribusi yang setara saat model belajar. 


Dengan menghilangkan perbedaan skala, model dapat lebih efektif dalam menemukan pola dalam data, mengurangi bias, dan meningkatkan performa prediksi. Hasil normalisasi menunjukkan bahwa semua nilai fitur kini terdistribusi dengan lebih merata, mendukung proses pembelajaran mesin yang lebih stabil dan akurat.


# Modeling
---

Pada proyek ini, beberapa model supervised learning diterapkan untuk tugas klasifikasi. Model-model tersebut meliputi:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)
5. Naive Bayes
6. Decision Tree

Keenam model machine learning di atas dibangun sekaligus dengan **parameter default** untuk melakukan klasifikasi:

1. **K-Nearest Neighbors (KNN)** [[4]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/)
   
   KNN adalah algoritma berbasis instance yang mencari sejumlah tetangga terdekat dari titik data dan memprediksi label berdasarkan mayoritas label tetangga
   tersebut.

    * Cara Kerja: Mengklasifikasikan data baru berdasarkan mayoritas dari "k" tetangga terdekatnya.
   * Kelebihan: Sederhana, tidak memerlukan asumsi distribusi data, bagus untuk dataset kecil.
   * Kekurangan: Lambat untuk dataset besar, sensitif terhadap noise.

   Default Parameter:
   * n_neighbors=5: Jumlah tetangga yang akan digunakan dalam prediksi.
   * metric='minkowski': Fungsi jarak yang digunakan, secara default menggunakan jarak Euclidean.
   * weights='uniform': Semua tetangga memiliki bobot yang sama.
2. **Random Forest** [[5]](https://repository.usd.ac.id/35513/)
   
   Random Forest adalah algoritma ensemble yang terdiri dari beberapa pohon keputusan (decision trees) yang dilatih pada subset acak dari data. Prediksi akhir
   didasarkan pada agregasi (voting mayoritas) hasil dari tiap pohon.

    * Cara Kerja: Menggunakan banyak pohon keputusan yang dilatih pada subset data yang berbeda dan menggabungkan hasilnya.
    * Kelebihan: Baik dalam menangani overfitting, bisa menangani banyak fitur.
    * Kekurangan: Relatif lambat untuk prediksi, sulit diinterpretasikan.

   
   Default Parameter:
   * n_estimators=100: Jumlah pohon dalam hutan.
   * criterion='gini': Digunakan untuk mengukur kualitas split.
   * max_depth=None: Tidak ada batasan pada kedalaman pohon.
     
3. **Support Vector Machine (SVM)** [[6](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)]
   
   SVM adalah algoritma klasifikasi yang mencari hyperplane terbaik yang memisahkan dua kelas dengan margin terbesar.

   * Cara Kerja: Mencari hyperplane yang memisahkan kelas dengan margin maksimum.
   * Kelebihan: Efektif untuk ruang berdimensi tinggi, menggunakan kernel trick untuk data non-linear.
   * Kekurangan: Tidak efisien untuk dataset besar, memerlukan scaling.

   Default Parameter:
   * C=1.0: Parameter regulasi untuk menghindari overfitting.
   * kernel='rbf': Menggunakan kernel Gaussian Radial Basis Function (RBF).
   * gamma='scale': Menentukan seberapa jauh pengaruh satu sampel individu.
     
4. **Naive Bayes (Gaussian Naive Bayes)** [[7](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)]
   
   Naive Bayes adalah algoritma berbasis probabilitas yang menghitung kemungkinan setiap kelas berdasarkan teorema Bayes dan asumsi bahwa fitur bersifat
   independen satu sama lain.

   * Cara Kerja: Menggunakan Teorema Bayes dengan asumsi independensi antar fitur.
   * Kelebihan: Cepat, efisien untuk dataset besar, baik untuk klasifikasi teks.
   * Kekurangan: Asumsi independensi sering tidak realistis.
 
   Default Parameter:
   * Asumsi distribusi Gaussian pada fitur yang kontinu.
     
5. **Decision Tree** [[8](https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948)]
   
   Decision Tree adalah algoritma yang membuat model berdasarkan aturan if-then dari fitur data untuk membuat prediksi. Setiap cabang pohon mewakili keputusan.

   * Cara Kerja: Membagi dataset berdasarkan kondisi untuk membuat keputusan.
   * Kelebihan: Mudah dipahami, tidak memerlukan scaling.
   * Kekurangan: Cenderung overfitting jika tidak dipangkas
       
   Default Parameter:
   * criterion='gini': Digunakan untuk mengukur kualitas split.
   * splitter='best': Pemilihan split terbaik di setiap simpul.
   * max_depth=None: Tidak ada batasan pada kedalaman pohon.
     
6. **XGBoost (Extreme Gradient Boosting)** [[9](https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948)]
   
   XGBoost adalah algoritma ensemble berbasis gradient boosting yang sangat efisien dan cepat, serta mampu menangani outlier dan missing values secara lebih baik
   dibandingkan model lainnya.

   * Cara Kerja: Algoritma boosting yang membangun banyak pohon keputusan untuk memperbaiki kesalahan sebelumnya.
   * Kelebihan: Efisien dalam waktu pelatihan, menangani missing data otomatis.
   * Kekurangan: Membutuhkan tuning hyperparameter yang rumit.
   
   Default Parameter:
   * n_estimators=100: Jumlah pohon boosting yang akan digunakan.
   * learning_rate=0.1: Kecepatan pembelajaran atau dampak dari setiap pohon.
   * max_depth=6: Kedalaman maksimum dari pohon keputusan.

## Evaluation
---

Untuk menentukan kinerja model, perlu untuk mengevaluasi model yang sudah dibangun. Model klasifikasi akan dievaluasi menggunakan kriteria evaluasi seperti akurasi, presisi (*precision*), *recall*, dan *f1-score*. Persamaan-persamaan berikut menunjukkan perhitungan untuk mendapatkan evaluasi model.	

- Akurasi adalah rasio prediksi yang benar terhadap jumlah estimasi secara keseluruhan. Rumus untuk menghitung akurasi ditunjukkan dalam Persamaan berikut:

  $$ Accuracy = {TP+TN \over(TP+TN+FP+FN)} $$
- Presisi (*Precision*) merupakan perbandingan antara jumlah prediksi positif yang tepat dengan keseluruhan hasil prediksi positif. Presisi dihitung menggunakan persamaan berikut:

$$ Precision = {TP \over(TP+FP)} $$
- *Recall* adalah perbandingan antara jumlah prediksi positif dengan jumlah data positif secara keseluruhan. *Recall* dihitung menggunakan persamaan berikut:

$$ Recall = {TP \over(TP+FN)} $$
- *F1-score* adalah suatu bentuk keseimbangan yang menggabungkan akurasi dan *recall* dalam sebuah sistem. Ini merupakan nilai rata-rata harmonis antara presisi dan *recall*. *F1-score* dihitung menggunakan persamaan berikut:

$$ F1-score = {2* precision*recall \over precision+ recall} $$


### Metrik Evaluasi

Metrik evaluasi yang digunakan dalam proyek ini adalah Accuracy, Precision, dan Recall. Hasil evaluasi menunjukkan perbedaan kinerja di antara model:

1. Accuracy: Mengukur persentase prediksi yang benar.
2. Precision: Mengukur ketepatan prediksi positif.
3. Recall: Mengukur kemampuan model dalam mendeteksi semua kasus positif.


Setelah proses modeling, akurasi yang didapatkan adalah sebagai berikut:

```python
                 Model  Accuracy  Precision    Recall
      0            KNN   0.89750   0.897644  0.897524
      1  Random Forest   0.90875   0.908766  0.908759
      2            SVM   0.90750   0.907532  0.907512
      3    Naive Bayes   0.76250   0.762737  0.762461
      4  Decision Tree   0.80125   0.801293  0.801264
      5        XGBoost   0.90625   0.906250  0.906253
```

#### Confusion Matrix:

<details>
<summary>K-Nearest Neighbors (KNN)</summary>

![KNN](https://i.postimg.cc/v8KNqQcP/KNN.png)

</details>

<details>
<summary>Random Forest</summary>

![RF](https://i.postimg.cc/4NbFJpW0/RF.png)

</details>

<details>
<summary>XGBoost</summary>

![XGBoost](https://i.postimg.cc/J0g70971/XG.png)

</details>

<details>
<summary>Support Vector Machine (SVM)</summary>

![SVM](https://i.postimg.cc/2jvC7X5B/svm.png)

</details>

<details>
<summary>Naive Bayes</summary>

![Naive Bayes](https://i.postimg.cc/PxyzgMXb/nv.png)

</details>

<details>
<summary>Decision Tree</summary>

![Decision Tree](https://i.postimg.cc/59ZqZj1R/DT.png)

</details>

### Hasil Evaluasi Model 

Tabel 3. Hasil Accuracy

![Plot Accuracy](https://i.postimg.cc/15GmLHJZ/akurasi.png)

Gambar 3. Visualisasi Accuracy Model

Grafik 3. membandingkan akurasi berbagai model klasifikasi yang telah diterapkan. XGBoost memiliki akurasi tertinggi (0.91), menunjukkan performa terbaik dalam memprediksi data. SVM, Random Forest, dan KNN semuanya menunjukkan akurasi yang sama (0.90), hanya sedikit lebih rendah dari XGBoost. Decision Tree memiliki akurasi 0.81, yang masih cukup baik, namun berada di bawah ketiga model sebelumnya. Naive Bayes mencatat akurasi terendah (0.76), yang berarti model ini kurang efektif dalam menangani dataset dibandingkan model lainnya.

Evaluasi model machine learning yang telah dilakukan menunjukkan bahwa model-model yang digunakan, seperti K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), Naive Bayes, Decision Tree, dan XGBoost, berhasil menjawab problem statement terkait prediksi kualitas apel secara lebih akurat dibandingkan dengan pendekatan konvensional. Dengan menggunakan metrik evaluasi seperti akurasi, presisi, dan recall, model-model ini mampu memberikan hasil yang signifikan dalam memprediksi kualitas buah berdasarkan fitur yang terukur.

Selain itu, model ini mencapai goals yang diharapkan, yaitu meningkatkan efisiensi dan akurasi dalam proses klasifikasi apel, yang sebelumnya bergantung pada metode manual dan analisis laboratorium yang memakan waktu dan tidak selalu akurat. Dengan implementasi model ini, produsen dapat membuat keputusan yang lebih baik dalam memilih apel untuk konsumsi atau penjualan, sehingga membantu meningkatkan kualitas produk dan kepuasan konsumen.

Dari segi dampak, solusi yang diusulkan tidak hanya memberikan keuntungan bagi produsen dalam hal penghematan waktu dan biaya, tetapi juga berpotensi meningkatkan daya saing di pasar, yang pada akhirnya dapat mendukung keberlanjutan bisnis dalam industri pertanian. Dengan demikian, evaluasi model ini tidak hanya menunjukkan kinerja teknis, tetapi juga relevansi dan kontribusi terhadap pemecahan masalah yang dihadapi dalam bisnis apel.

### Kesimpulan

Model machine learning yang kami bangun berhasil meningkatkan akurasi prediksi kualitas apel, yang lebih efisien dibandingkan metode konvensional. Model ini telah mencapai tujuan utama kami dalam membantu produsen mengambil keputusan yang lebih baik dan meningkatkan daya saing di pasar. Dengan demikian, solusi yang kami usulkan memiliki dampak positif terhadap bisnis, menghemat waktu dan biaya, serta meningkatkan kualitas produk dan kepuasan konsumen. Meskipun hasil yang diperoleh memuaskan, masih ada ruang untuk pengembangan lebih lanjut dengan menggunakan dataset yang lebih besar atau fitur tambahan untuk meningkatkan akurasi.

## Referensi
--- 
 
[1] Sellitasari, Shelvi., Ainurrasyid., & Suryanto, Agus. (2013). _PERBEDAAN PRODUKSI TANAMAN APEL (Malus sylvestris mill.) PADA AGROKLIMAT YANG BERBEDA (Studi Kasus Pada Sentra Produksi Tanaman Apel di Kota Batu dan Kabupaten Malang)_. Tersedia: [tautan](https://protan.studentjournal.ub.ac.id/index.php/protan/article/view/1). Diakses pada 24 Oktober 2024.

[2] Huang et al. (2018). _Applications of Support Vector Machine (SVM) Learning in Cancer Genomics_. Tersedia: [tautan](https://cgp.iiarjournals.org/content/15/1/41.abstract). Diakses pada 24 Oktober 2024.

[3] Ridwan, Ahmad. (2020). _Penerapan Algoritma Naïve Bayes Untuk Klasifikasi Penyakit Diabetes Mellitus_. Tersedia: [tautan](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/169). Diakses pada 24 Oktober 2024.

[4] Sani, Ramadhan Rakhmat.,  Zeniarja, Junta., & Luthfiarta, Ardytha. (2016). _Penerapan Algoritma K-Nearest Neighbor pada Information Retrieval dalam Penentuan Topik Referensi Tugas Akhir_. Tersedia: [tautan](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/). Diakses pada 21 Oktober 2024.

[5] Haristu, Reinardus Aji (2019) _Penerapan metode Random Forest untuk prediksi win ratio pemain player Unknown Battleground_. Tersedia: [tautan](https://repository.usd.ac.id/35513/). Diakses pada 21 Oktober 2024.

[6] Gandhi, Rohits. (2018). _Support Vector Machine — Introduction to Machine Learning Algorithms_. Tersedia: [tautan](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47). Diakses pada 23 Oktober 2024.

[7] Gandhi, Rohits. (2018). _Naive Bayes Classifier_. Tersedia: [tautan](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47). Diakses pada 23 Oktober 2024.

[8] Gandhi, Rohits. (2018). _Decision Tree, Random Forest, and XGBoost: An Exploration into the Heart of Machine Learning_. Tersedia: [tautan](https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948). Diakses pada 23 Oktober 2024.
