# Laporan Proyek Machine Learning - Zahwa Genoveva
---
## Domain Proyek
---
Domain proyek yang dipilih dalam proyek _machine learning_ ini adalah mengenai Pertanian dengan judul proyek "Prediksi kualitas buah apel".

* #####  Latar Belakang
  ![](https://cdn1-production-images-kly.akamaized.net/xBRT773xJ9vvZx12J6fN1IIVhko=/1200x675/smart/filters:quality(75):strip_icc():format(jpeg)/kly-media-production/medias/2792644/original/076377400_1556614591-aaron-blanco-tejedor-390113-unsplash.jpg)
Apel (Malus domestica) adalah salah satu jenis buah yang sangat diminati masyarakat berkat variasi rasa yang ditawarkannya. Di industri pertanian dan pemasaran, kualitas buah apel menjadi faktor penting yang sangat memengaruhi nilai jual dan minat konsumen [[1]](https://doi.org/10.30606/rjocs.v10i2.2856). Untuk itu, berbagai metode telah dikembangkan guna memprediksi dan meningkatkan kualitas buah ini. Meskipun pendekatan konvensional seperti pengukuran manual dan analisis laboratorium masih sering digunakan, metode ini cenderung membutuhkan waktu yang lama serta tidak selalu memberikan hasil yang akurat.

  Seiring dengan pesatnya perkembangan teknologi, khususnya di bidang machine learning, potensi untuk meningkatkan prediksi kualitas buah apel semakin terbuka lebar. Teknik-teknik analisis prediktif berbasis machine learning dapat membantu mengungkap pola-pola kompleks dalam data yang sulit diidentifikasi menggunakan metode konvensional [[2]](https://doi.org/10.21873/cgp.20063).. Lebih jauh, penerapan data preprocessing dalam pengolahan dataset memiliki peran penting dalam mengoptimalkan kualitas data, meningkatkan akurasi model, serta mengurangi bias yang mungkin timbul dalam proses prediksi [[3]](https://doi.org/10.47970/siskom-kb.v4i1.169).

## Business Understanding
---
#### Problem Statements
berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
* Bagaimana cara melakukan pra-pemrosesan pada data prediksi apel yang akan digunakan untuk membuat model yang baik?
* Bagaimana cara mengatasi masalah data yang tidak seimbang pada dataset kualitas apel?
* Berapa nilai akurasi terbaik yang didapatkan dengan menggunakan _machine learning_?
* Algoritma machine learning mana yang paling efektif untuk menyelesaikan masalah klasifikasi kualitas apel?       


#### Goals
* Melakukan pra-pemrosesan dengan baik agar dapat digunakan dalam pembuatan model.
* Mengetahui cara membuat model machine learning untuk memprediksi kualitas buah apel.
* Membuat model _machine learning_ dengan nilai akurasi yang mencapai 90%.

#### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
* Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Melakukan _drop_ kolom pada kolom ID.
    * Mengatasi masalah data yang kosong dengan nilai rata-rata kolom (_mean substitution_).
    * Melakukan Encoding terhadap kolom yang bertipe _object_.
    * Mengatasi masalah data tidak seimbang dengan _resample_.
    * Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
    * Melakukan _Standard Scaler_.

* Untuk pembuatan model dipilih penggunaan model dengan algoritma Random Forest dan K-Nearest Neighbor. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. Berikut cara kerja, kelebihan dan kekurangan dari masing-masing algoritma yang digunakan
  1. K-Nearest Neighbors (KNN) [[4]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/)
     * Cara Kerja: Mengklasifikasikan data baru berdasarkan mayoritas dari "k" tetangga terdekatnya.
     * Kelebihan: Sederhana, tidak memerlukan asumsi distribusi data, bagus untuk dataset kecil.
     * Kekurangan: Lambat untuk dataset besar, sensitif terhadap noise.
  2. Random Forest [[5]](https://repository.usd.ac.id/35513/)
     * Cara Kerja: Menggunakan banyak pohon keputusan yang dilatih pada subset data yang berbeda dan menggabungkan hasilnya.
     * Kelebihan: Baik dalam menangani overfitting, bisa menangani banyak fitur.
     * Kekurangan: Relatif lambat untuk prediksi, sulit diinterpretasikan.
  3. Support Vector Machine (SVM) [[6](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)]
     * Cara Kerja: Mencari hyperplane yang memisahkan kelas dengan margin maksimum.
     * Kelebihan: Efektif untuk ruang berdimensi tinggi, menggunakan kernel trick untuk data non-linear.
     * Kekurangan: Tidak efisien untuk dataset besar, memerlukan scaling.
  4. Naive Bayes [[7](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)]
     * Cara Kerja: Menggunakan Teorema Bayes dengan asumsi independensi antar fitur.
     * Kelebihan: Cepat, efisien untuk dataset besar, baik untuk klasifikasi teks.
     * Kekurangan: Asumsi independensi sering tidak realistis.
  5. Decision Tree [[8](https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948)]
     * Cara Kerja: Membagi dataset berdasarkan kondisi untuk membuat keputusan.
     * Kelebihan: Mudah dipahami, tidak memerlukan scaling.
     * Kekurangan: Cenderung overfitting jika tidak dipangkas
  6. XGBoost [[9](https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948)]
     * Cara Kerja: Algoritma boosting yang membangun banyak pohon keputusan untuk memperbaiki kesalahan sebelumnya.
     * Kelebihan: Efisien dalam waktu pelatihan, menangani missing data otomatis.
     * Kekurangan: Membutuhkan tuning hyperparameter yang rumit.
    
## Data Understanding
![Image of Dataset](https://i.postimg.cc/CKJ0sBXT/Screenshot-2024-10-10-204041.png)
Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data)
Lisensi | Data files © Original Authors
Kategori | Computer Science, Education, Food, Data Visualization, Classification, Exploratory Data Analysis
Jenis dan Ukuran Berkas | CSV (387.65 kB)
---

Pada tahap ini, kita akan menganalisis struktur dan karakteristik dataset yang digunakan. Dataset terdiri dari 9 kolom dan 4001 baris, dengan rincian sebagai berikut:

###  1. Deskripsi Data
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

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| 0.0 | -3.970049 |-2.512336 | 5.346330 |-1.012009 | 1.844900 |0.329840	| -0.491590483  |good |
| 1.0 | -1.195217 |-2.839257 | 3.664059 |1.588232 | 0.853286 | 0.867530 | -0.722809367  |good |
| 2.0 | -0.292024 |	-1.351282 | -1.738429 | -0.342616 | 2.838636 |-0.038033	| 2.621636473  |bad |
| 3.0 | -0.657196 |-2.271627 | 1.324874 |-0.097875 | 3.637970 |-3.413761	| 0.790723217  |good |
| 4.0 | 1.364217 |-1.296612 | -0.384658 | -0.553006 | 3.030874 | -1.303849	| 0.501984036  |good |
  


### 3. Menangani Missing Values

Untuk mendeteksi missing value digunakan fungsi isnull().sum() dan diperoleh: **Tabel 1. Hasil Deteksi Missing Value**
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

### 4. Menangani Outliers

Untuk menangani outliers, metode **Interquartile Range (IQR)** digunakan. IQR mengidentifikasi data yang berada di luar rentang normal, yaitu di luar batas [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

**Tabel 2 Visualisasi Boxplot Sebelum dan Sesudah Metode IQR**:

| Boxplot Sebelum IQR        | Boxplot Setelah IQR        |
|----------------------------|----------------------------|
| ![Before](https://i.postimg.cc/BQ72GNWp/sebelum.png)                | ![After](https://i.postimg.cc/X7PBPJFD/sesudah.png)                |


### 5. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outliers

Setelah dilakukan pembersihan outliers dengan metode **IQR**, jumlah data yang tersisa dibandingkan dengan jumlah data awal adalah seperti pada **Tabel 3. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outlier**:

| Jumlah Data Sebelum  | Jumlah Data Setelah |
|----------------------|---------------------|
| 4001          |           3790          |




## Univariate Analysis
 
Selanjutnya, akan dilakukan proses analisis data dengan teknik Univariate EDA. Pada kasus ini semua fiturnya adalah fitur numerik dan tidak ada fitur kategorikal. Sehingga hanya perlu dilakukan analisa terhadap fitur numerik, sebagai berikut:

### Analisa Fitur Numerik

Untuk melihat distribusi data pada tiap fitur akan digunakan visualisasi dengan histogram sebagai berikut:

![histogram](https://i.postimg.cc/CxqtBcgM/ultra.png)

Dari hasil visualisasi histogram di atas, kita bisa memperoleh beberapa informasi, antara lain:

Distribusi fitur quality (target) cenderung miring ke kanan (right-skewed).

Karena beberapa fitur belum terdistribusi normal hal ini akan berimplikasi pada model, maka selanjutnya kita lakukan transformasi data (non-linear scaling). Namun, sebelum itu kita cek terlebih dahulu hubungan antara fitur numerik tersebut.

## Multivariate Analysis

### Hubungan Antara Fitur Numerik

Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut:

![grafik_pairplot](https://i.postimg.cc/ncbjvz84/multi.png)

Gambar 2. Visualisasi Hubungan antara Fitur Numerik dengan pairplot()

Secara keseluruhan, pairplot ini membantu kita memahami kompleksitas hubungan antara berbagai karakteristik (seperti ukuran, berat, kemanisan, dsb.) dari suatu objek. Kita bisa melihat variabel mana yang saling terkait, variabel mana yang tidak, dan variabel mana yang mungkin mempengaruhi kualitas.

### Korelasi antara Fitur Numerik

Untuk mengevaluasi skor korelasi hubungan antara fitur numerik, akan digunakan fungsi corr() dengan output sebagai berikut.

![matriks](https://i.postimg.cc/25wFtWYN/matriks.png)

Gambar 3. Korelasi antara Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Berdasarkan matriks korelasi ini, kita bisa menyimpulkan bahwa kualitas suatu objek dipengaruhi oleh beberapa faktor seperti ukuran, juiciness, acidity, sweetness, dan ripeness. Objek yang lebih besar, lebih juicy, lebih asam, dan kurang manis serta kurang matang cenderung memiliki kualitas yang lebih baik.


---

Bagian ini menunjukkan proses eksplorasi data yang menyeluruh, mulai dari penanganan missing values hingga outliers, yang dilakukan sebelum melanjutkan ke tahap pemodelan machine learning.


## Data Preparation
---
Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data:

## Train Test Split

Dataset akan dibagi menjadi data latih (train) dan data uji (test). Tujuan langkah ini sebelum proses lainnya adalah agar tidak mengotori data uji dengan informasi yang didapat dari data latih. Contoh pada proses standarisasi dimana jika belum di bagi menjadi data latih dan uji, maka keduanya akan terkena transformasi data yang menggunakan informasi (mean dan standard deviation) dari gabungan data latih dan uji. Hal ini berpotensi menimbulkan kebocoran data (data leakage). Oleh karena itu langkah awal sebelum melakukan tranformasi data adalah membagi dataset terlebih dahulu [3].

Pada kasus ini akan menggunakan proporsi pembagian sebesar 90:10 dengan fungsi train_test_split dari sklearn dengan output sebagai berikut.

Tabel 4. Jumlah Data Latih dan Uji

Jumlah Total Data | Jumlah Data Latih | Jumlah Data Uji
----------------- | ----------------- | ---------------
  4000 | 3200 | 800

## Standarisasi
Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada kasus ini akan digunakan metode StandarScaler() dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.

Berikut output yang dihasilkan dari metode StandardScaler dengan menggunakan fungsi describe():

Tabel 5. Hasil Proses Standarisasi Pada Setiap Fitur Pada Data Latih
### 2. Standarisasi (Normalisasi Fitur)

Pada langkah ini, kita akan menstandarisasi fitur dengan menggunakan `StandardScaler`, sehingga setiap fitur memiliki rata-rata 0 dan deviasi standar 1.

| Deskripsi                             | Keterangan                              |
|---------------------------------------|-----------------------------------------|
| Metode Standarisasi                   | StandardScaler                          |
| Rata-rata fitur setelah standarisasi   | 0                                       |
| Deviasi standar fitur setelah standarisasi | 1                                       |

### 3. Normalisasi

Selain standarisasi, kita juga menerapkan normalisasi menggunakan `MinMaxScaler`, yang mengubah skala fitur ke dalam rentang [0, 1].

#### Hasil Normalisasi:

|     Size     |    Weight    |  Sweetness  |  Crunchiness |  Juiciness   |  Ripeness   |   Acidity   |
|--------------|--------------|-------------|--------------|--------------|-------------|-------------|
|  1.027890    | -1.001988    |  0.084334   |  0.393453    |  0.271185    |  1.108047   | -0.627140   |
|  0.173070    |  0.852012    | -0.323051   | -0.354721    | -0.106767    |  0.265260   | -1.075288   |
| -1.041694    | -1.165875    | -0.192792   | -1.034102    | -0.324381    |  2.470441   | -0.901013   |
| -0.456591    |  0.285175    |  1.948610   | -0.588818    |  1.024831    | -0.509328   | -0.057273   |
| -0.816011    | -0.146890    | -0.177478   |  2.012460    |  0.249208    |  0.139939   | -0.030334   |

### 4. Perbandingan Nilai Asli dan Nilai Normalisasi

| Original Size | Original Weight | Original Sweetness | Original Crunchiness | Original Juiciness | Original Ripeness | Original Acidity | Normalized Size | Normalized Weight | Normalized Sweetness | Normalized Crunchiness | Normalized Juiciness | Normalized Ripeness | Normalized Acidity |
|---------------|------------------|---------------------|----------------------|--------------------|--------------------|-------------------|------------------|---------------------|----------------------|-----------------------|---------------------|---------------------|---------------------|
|  1.027890     | -1.001988        |  0.084334           |  0.393453            |  0.271185          |  1.108047          | -0.627140         |  1.027890        | -1.001988           |  0.084334            |  0.393453             |  0.271185           |  1.108047           | -0.627140           |
|  0.173070     |  0.852012        | -0.323051           | -0.354721            | -0.106767          |  0.265260          | -1.075288         |  0.173070        |  0.852012           | -0.323051            | -0.354721             | -0.106767           |  0.265260           | -1.075288           |
| -1.041694     | -1.165875        | -0.192792           | -1.034102            | -0.324381          |  2.470441          | -0.901013         | -1.041694        | -1.165875           | -0.192792            | -1.034102            | -0.324381           |  2.470441           | -0.901013           |
| -0.456591     |  0.285175        |  1.948610           | -0.588818            |  1.024831          | -0.509328          | -0.057273         | -0.456591        |  0.285175           |  1.948610            | -0.588818            |  1.024831           | -0.509328           | -0.057273           |
| -0.816011     | -0.146890        | -0.177478           |  2.012460            |  0.249208          |  0.139939          | -0.030334         | -0.816011        | -0.146890           | -0.177478            |  2.012460            |  0.249208           |  0.139939           | -0.030334           |

### 5. Data setelah Normalisasi

|     Size     |    Weight    |  Sweetness  |  Crunchiness |  Juiciness   |  Ripeness   |   Acidity   |
|--------------|--------------|-------------|--------------|--------------|-------------|-------------|
|  0.63683187  |  0.35305009  |  0.49645027 |  0.55451442  |  0.52664718  |  0.64304288 |  0.40105263 |
|  0.51523605  |  0.58231483  |  0.43646357 |  0.47743345  |  0.47180237  |  0.52263894 |  0.33565786 |
|  0.34243895  |  0.33278396  |  0.45564396 |  0.40744007  |  0.44022423  |  0.83767975 |  0.36108855 |
|  0.42566827  |  0.51222009  |  0.77096161 |  0.45331553  |  0.63600913  |  0.41197840 |  0.48420897 |
|  0.37454175  |  0.45879104  |  0.45789899 |  0.72131318  |  0.52345814  |  0.50473515 |  0.48814003 |


Normalisasi merupakan proses penting dalam tahap persiapan data, yang bertujuan untuk mengubah skala fitur agar berada dalam rentang yang sama. Dalam konteks dataset ini, normalisasi membantu memastikan bahwa setiap atribut, seperti Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, dan Acidity, memiliki kontribusi yang setara saat model belajar. Dengan menghilangkan perbedaan skala, model dapat lebih efektif dalam menemukan pola dalam data, mengurangi bias, dan meningkatkan performa prediksi. Hasil normalisasi menunjukkan bahwa semua nilai fitur kini terdistribusi dengan lebih merata, mendukung proses pembelajaran mesin yang lebih stabil dan akurat.



# Modeling
Pada tahap ini, akan menggunakan tiga algoritma untuk regresi. Kemudian, akan dilakukan evaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:

1. K-Nearest Neighbor
