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
Sumber | [Kaggle Dataset : Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
Lisensi | Data files © Original Authors
Kategori | Kesehatan, Kondisi Kesehatan, Kesehatan Masyarakat
Jenis dan Ukuran Berkas | CSV (316.97 kB)
---
Pada berkas yang diunduh yakni healthcare-dataset-stroke-data.csv berisi 5110 baris dan 12 kolom. Kolom-kolom tersebut terdiri dari 5 buah kolom bertipe objek dan 7 buah kolom bertipe numerik (tipe data int64). Terdapat juga kolom yang memiliki data kosong yaitu pada kolom bmi. Untuk penjelasan mengenai variabel-variable pada dataset stroke ini dapat dilihat sebagai berikut:
* **id** merupakan parameter bernilai unique. Parameter ini tidak penting untuk dimasukkan kedalam model, oleh karena itu parameter ini di _drop_.
* **gender** merupakan parameter untuk mengetahui jenis kelamin. Terdapat 3 nilai yaitu _male_, _female_, dan _other_.
* **age** merupakan parameter untuk mengetahui umur. Terdapat Pada data ini nilainya berada pada rentang 0.080-82 tahun.
* **hypertension** merupakan parameter yang menyatakan apakah pasien memiliki darah tinggi atau tidak. Nilai 0 menyatakan bahwa pasien tidak memiliki darah tinggi dan Nilai 1 menyatakan bahwa pasien memiliki darah tinggi.
* **heart_disease** merupakan parameter yang menyatakan apakah pasien memiliki penyakit jantung atau tidak. Nilai 0 menyatakan bahwa pasien tidak memiliki penyakit jantung dan Nilai 1 menyatakan bahwa pasien memiliki penyakit jantung.
* **ever_married** merupakan parameter yang menyatakan apakah pasien pernah menikah atau tidak. Nilai "_Yes_" menyatakan bahwa pasien pernah menikah dan Nilai "_No_" menyatakan bahwa pasien belum pernah menikah.
* **work_type** merupakan parameter yang menyatakan pekerja pasien. Pada data ini terdapat 5 nilai yaitu "_children_", "_Govt_jov_", "_Never_worked_", "_Private_" dan "_Self-employed_".
* **Residence_type** merupakan parameter yang menyatakan tipe tempat tinggal pasien. Pada data ini terdapat 2 nilai yaitu "_Rural_" dan "_Urban_".
* **avg_glucose_level** merupakan parameter yang menyatakan kadar glukosa rata-rata dalam darah pasien. Pada data ini nilainya berada di rentang 55.12-271.74 mg/dL.
* **bmi** merupakan parameter yang menyatakan kadar glukosa rata-rata dalam darah pasien. Pada data ini nilainya berada di rentang 55.12-271.74.
* **smoking_status** merupakan parameter yang menyatakan status merokok pada pasien. Pada data ini terdapat 4 nilai yaitu "_formerly smoked_", "_never smoked_", "_smokes_" or "_Unknown_".
* **stroke** merupakan parameter yang Menentukan apakah pasien menderita stroke atau tidak. Terdapat 2 nilai yaitu tidak menderita stroke (nilai 0) dan menderita stroke (nilai 1).

Selain itu, terdapat juga visualisasi data pada tiap kolom yang dibagi menjadi 2 tipe seperti berikut:
* Kategorial:
