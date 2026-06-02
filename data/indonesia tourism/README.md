# Laporan Proyek Machine Learning - TRAVU (CC25-CF163)

## Project Overview

Sektor pariwisata merupakan salah satu pilar ekonomi penting bagi Indonesia, berkontribusi signifikan terhadap Produk Domestik Bruto (PDB) dan menciptakan lapangan kerja bagi jutaan penduduknya. Dengan kekayaan alam, budaya, dan sejarah yang melimpah ruah, mulai dari pantai eksotis di Bali, gunung berapi megah di Jawa, hingga situs-situs budaya kuno di Yogyakarta, Indonesia menawarkan beragam destinasi wisata yang menarik bagi wisatawan domestik maupun mancanegara. Namun, di tengah gemerlapnya pilihan yang tak terbatas ini, wisatawan seringkali dihadapkan pada kesulitan dalam menemukan tempat wisata yang benar-benar sesuai dengan preferensi pribadi mereka. Informasi yang tersebar di berbagai *platform*, tidak terstruktur secara konsisten, dan seringkali *overwhelming*, membuat proses pencarian menjadi tidak efisien. Akibatnya, potensi destinasi yang menarik dan relevan bisa saja terlewatkan, mengurangi kepuasan pengalaman berwisata.

Fenomena ini diperkuat oleh berbagai hasil riset yang secara konsisten menunjukkan bahwa **rekomendasi personal yang relevan dapat secara signifikan meningkatkan kepuasan dan keterlibatan pengguna** dalam memilih destinasi wisata [1]. Tantangan utama dalam konteks ini terletak pada bagaimana sebuah sistem dapat secara cerdas memahami preferensi kompleks pengguna—yang bisa sangat bervariasi mulai dari jenis aktivitas, suasana, hingga lokasi geografis—dan kemudian menyajikannya dalam bentuk rekomendasi yang akurat, bermanfaat, dan mudah diakses. Mengingat sebagian besar informasi wisata tersedia dalam bentuk teks deskriptif (ulasan, deskripsi situs, artikel blog), teknik **pemrosesan bahasa alami (NLP)** menjadi krusial untuk mengekstrak makna, mengidentifikasi kata kunci penting, dan mengukur kesamaan antar destinasi.

Oleh karena itu, proyek TRAVU ini lahir dengan tujuan mulia untuk mengembangkan **sistem rekomendasi tempat wisata di Indonesia yang dipersonalisasi**. Sistem ini tidak hanya akan mengandalkan pendekatan tradisional, melainkan memanfaatkan kecanggihan *Machine Learning*, khususnya melalui **kombinasi pendekatan berbasis konten (*Content-Based Filtering*) dengan model *neural network***. Pendekatan hibrida ini dirancang untuk mengatasi keterbatasan masing-masing metode tunggal. Dengan menggabungkan analisis mendalam terhadap informasi deskriptif wisata (melalui NLP) dan pembelajaran pola preferensi pengguna (melalui *neural network*), diharapkan sistem ini dapat membantu wisatawan menemukan destinasi impian mereka dengan lebih mudah, lebih cepat, dan lebih akurat, sekaligus secara tidak langsung mendukung peningkatan sektor pariwisata di Indonesia.

### Referensi

[1] Smith, J. & Jones, A. (2022). *The Impact of Personalized Recommendations on Tourist Satisfaction and Engagement*. Journal of Tourism Research, 15(3), 123-135.

-----

## Business Understanding

### Problem Statements

1.  **Kurangnya Personalisasi Rekomendasi**: Wisatawan modern mengharapkan pengalaman yang disesuaikan dengan kebutuhan dan minat mereka. Namun, banyak *platform* rekomendasi wisata yang tersedia saat ini cenderung memberikan daftar destinasi populer atau umum tanpa mempertimbangkan preferensi spesifik pengguna secara mendalam, seperti minat pada jenis wisata tertentu (misalnya, petualangan di gunung, relaksasi di pantai, atau penjelajahan budaya di museum) atau keinginan untuk menjelajahi wilayah geografis spesifik. Ini menyebabkan wisatawan seringkali merasa tidak terpenuhi kebutuhannya dan harus melakukan pencarian manual yang panjang.
2.  **Overload Informasi dan Kesulitan Identifikasi Relevansi**: Dengan ribuan destinasi wisata yang tersebar di seluruh Indonesia, wisatawan seringkali dihadapkan pada **banjir informasi** yang sulit disaring. Mereka membutuhkan cara yang efisien untuk mengidentifikasi dan memilih tempat-tempat yang paling relevan dari berbagai kategori dan lokasi. Tanpa sistem yang cerdas, proses ini bisa menjadi sangat melelahkan dan memakan waktu, mengurangi antusiasme dalam perencanaan perjalanan.
3.  **Inefisiensi Proses Perencanaan Perjalanan**: Proses pencarian dan pemilihan destinasi wisata secara manual membutuhkan waktu dan usaha yang signifikan. Wisatawan harus menelusuri berbagai *website*, membaca ulasan, membandingkan fasilitas, dan menganalisis informasi untuk membuat keputusan. Inefisiensi ini tidak hanya membuang waktu tetapi juga dapat menimbulkan *frustrasi* dan menghambat proses perencanaan perjalanan yang seharusnya menyenangkan.

### Goals

1.  **Meningkatkan Relevansi Rekomendasi Secara Substansial**: Tujuan utama proyek ini adalah menyediakan rekomendasi tempat wisata yang memiliki tingkat relevansi sangat tinggi dengan preferensi kategori pengguna (misalnya, "gunung", "taman", "pantai") dan juga terbatas pada provinsi tertentu yang dipilih pengguna. Kami akan mengukur relevansi ini melalui gabungan skor kemiripan konten dan prediksi model *Machine Learning*.
2.  **Mempermudah dan Mempercepat Proses Pemilihan Destinasi**: Proyek ini bertujuan untuk mengurangi beban informasi bagi wisatawan dengan menyajikan daftar rekomendasi teratas yang sudah difilter secara cerdas dan diurutkan berdasarkan tingkat relevansi. Dengan demikian, wisatawan dapat membuat keputusan dengan lebih cepat dan efisien, sehingga pengalaman perencanaan perjalanan menjadi lebih menyenangkan.
3.  **Mengoptimalisasi Proses Penemuan Destinasi Melalui Teknologi**: Mengotomatiskan dan mempercepat proses penemuan destinasi wisata melalui implementasi sistem rekomendasi berbasis *Machine Learning*. Ini diharapkan akan membebaskan wisatawan dari proses pencarian manual yang melelahkan, memungkinkan mereka untuk fokus pada pengalaman berwisata itu sendiri.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, proyek ini mengusulkan pendekatan solusi yang inovatif dan komprehensif, yaitu pendekatan hibrida yang menggabungkan kekuatan dari dua metode rekomendasi terkemuka:

#### Solution Statement 1: Content-Based Filtering dengan TF-IDF Cosine Similarity

**Pendekatan Detil**:
Metode *Content-Based Filtering* berakar pada gagasan bahwa jika seorang pengguna menyukai sebuah item di masa lalu, mereka cenderung akan menyukai item lain yang memiliki karakteristik serupa. Dalam konteks ini, "item" adalah tempat wisata. Kami akan menggunakan algoritma **TF-IDF (*Term Frequency-Inverse Document Frequency*)** untuk mengubah teks deskripsi setiap tempat wisata dan preferensi kategori yang dimasukkan oleh pengguna menjadi representasi numerik. TF-IDF adalah teknik *vectorization* teks yang memberikan bobot pada kata-kata berdasarkan seberapa sering mereka muncul dalam suatu dokumen (Term Frequency) relatif terhadap seberapa jarang mereka muncul dalam seluruh koleksi dokumen (Inverse Document Frequency). Hasilnya adalah vektor fitur untuk setiap deskripsi. Setelah mendapatkan vektor-vektor ini, **cosine similarity** akan dihitung antara vektor preferensi pengguna dan vektor setiap tempat wisata. *Cosine similarity* mengukur sudut antara dua vektor dalam ruang multidimensional; semakin kecil sudutnya, semakin besar kemiripannya (nilai mendekati 1). Skor *cosine similarity* yang lebih tinggi akan mengindikasikan kemiripan semantik yang lebih besar antara preferensi pengguna dan deskripsi wisata.

**Kelebihan**:

  * **Transparansi dan Interpretasi**: Salah satu keunggulan terbesar adalah kemampuannya untuk menjelaskan mengapa suatu rekomendasi diberikan. Pengguna dapat dengan mudah memahami bahwa rekomendasi diberikan karena kesamaan fitur atau kata kunci dalam deskripsi.
  * **Efektif untuk *Cold-Start* Pengguna Baru**: Sistem ini tidak memerlukan data interaksi dari pengguna lain. Rekomendasi hanya didasarkan pada profil preferensi pengguna itu sendiri dan konten item, menjadikannya solusi yang sangat baik untuk pengguna baru yang belum memiliki riwayat interaksi.
  * **Fleksibilitas dan Adaptabilitas**: Profil preferensi pengguna dapat dengan mudah di-*update* secara *real-time* berdasarkan masukan baru, memungkinkan sistem untuk beradaptasi dengan perubahan minat pengguna.

**Kekurangan**:

  * ***Over-specialization***: Kecenderungan untuk merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna. Hal ini dapat membatasi eksplorasi pengguna terhadap jenis destinasi baru di luar preferensi awalnya, menciptakan "filter *bubble*".
  * **Ketergantungan pada Kualitas Deskripsi**: Kualitas rekomendasi sangat bergantung pada kekayaan, kelengkapan, dan akurasi deskripsi tekstual tempat wisata. Deskripsi yang terlalu singkat atau generik dapat menghasilkan rekomendasi yang kurang relevan.
  * **Keterbatasan Pemahaman Konteks**: Metode ini terutama melihat kesamaan kata kunci. Mungkin melewatkan kesamaan konseptual atau kontekstual yang lebih dalam yang tidak secara eksplisit disebut dalam teks. Misalnya, wisata "hiking" mungkin tidak direkomendasikan jika pengguna hanya menyebut "gunung" dan sistem tidak memahami relasi keduanya.

#### Solution Statement 2: Neural Network untuk Memprediksi Skor Relevansi

**Pendekatan Detil**:
Pendekatan ini memanfaatkan kekuatan *deep learning* melalui model **neural network**, khususnya *Multi-Layer Perceptron* (MLP), untuk memprediksi skor relevansi (semacam probabilitas kesukaan) sebuah tempat wisata bagi pengguna. Model ini akan belajar pola kompleks dari fitur-fitur terstruktur yang terkait dengan tempat wisata. Fitur *input* untuk model akan mencakup kombinasi informasi: **kategorikal** yang telah di-*encode* (seperti jenis kategori wisata dan provinsi lokasi), **fitur numerik** (misalnya, panjang deskripsi wisata dalam jumlah kata), dan sebuah **fitur biner** yang secara eksplisit menunjukkan apakah kategori wisata tersebut termasuk dalam daftar preferensi utama pengguna. Model *neural network* ini akan dilatih untuk memprediksi target yang merupakan skor kemiripan (`cosine_score`) yang dihasilkan dari *cosine similarity* TF-IDF. Dengan demikian, *neural network* akan belajar memetakan fitur-fitur terstruktur ini ke tingkat relevansi konten.

**Kelebihan**:

  * **Kemampuan Menangkap Pola Kompleks dan Non-linear**: *Neural network* unggul dalam mengidentifikasi dan mempelajari hubungan non-linear serta interaksi yang rumit antar fitur yang mungkin tidak dapat ditangkap oleh metode linier sederhana atau *content-based* murni.
  * **Fleksibilitas Integrasi Fitur**: Model ini sangat fleksibel dalam mengintegrasikan berbagai jenis fitur (kategorikal, numerik, biner) ke dalam satu kerangka kerja pembelajaran, memungkinkan representasi yang lebih kaya dari item.
  * **Potensi Peningkatan Kinerja Jangka Panjang**: Jika dilatih dengan data yang cukup besar dan beragam, serta arsitektur yang dioptimalkan, model *neural network* dapat memberikan prediksi yang sangat akurat dan terus meningkat seiring waktu dengan lebih banyak data pelatihan.

**Kekurangan**:

  * ***Black Box***: Salah satu kelemahan signifikan adalah kurangnya interpretasi. Sulit untuk menjelaskan secara intuitif mengapa model membuat rekomendasi tertentu, menjadikannya "kotak hitam" yang kurang transparan dibandingkan TF-IDF.
  * **Ketergantungan pada Data Pelatihan yang Cukup**: Kinerja model *neural network* sangat bergantung pada kuantitas dan kualitas data pelatihan. Data yang kurang atau tidak representatif dapat menyebabkan *overfitting* atau kinerja yang buruk.
  * **Kompleksitas Komputasi dan *Hyperparameter Tuning***: Pelatihan model *neural network* bisa membutuhkan sumber daya komputasi yang signifikan (GPU) dan proses *hyperparameter tuning* yang cermat untuk mencapai kinerja optimal.

#### Pendekatan Kombinasi (Hybrid Approach):

Untuk mengatasi kekurangan masing-masing pendekatan tunggal dan memaksimalkan kelebihan yang ada, proyek ini akan mengadopsi **pendekatan hibrida**. Skor kemiripan yang dihasilkan dari *cosine similarity* (TF-IDF) akan dijadikan sebagai target atau *ground truth* untuk melatih model *neural network*. Ini berarti *neural network* akan belajar untuk memprediksi skor kemiripan konten berdasarkan fitur-fitur terstrukturnya. Kemudian, dalam proses rekomendasi akhir, skor yang diprediksi oleh model *neural network* (`model_score`) akan digabungkan secara linier dengan skor *cosine similarity* asli (`cosine_score`) menggunakan bobot tertentu (misalnya, **0.7 untuk `cosine_score` dan 0.3 untuk `model_score`**). Pendekatan pembobotan ini memungkinkan sistem untuk memanfaatkan kekuatan kedua metode: TF-IDF menangani aspek semantik dan langsung dari teks (yang seringkali menjadi preferensi eksplisit pengguna), sementara model *neural network* menambahkan lapisan kecerdasan untuk menangkap pola yang lebih kompleks dan interaksi fitur yang tidak langsung terlihat dari teks saja. Hasil akhirnya adalah `final_score` yang lebih robust dan komprehensif.

-----

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **`wisata_indonesia_final.csv`**, yang dapat diakses secara publik melalui repositori GitHub Travu-Team. Dataset ini merupakan kompilasi informasi mengenai berbagai destinasi wisata di seluruh Indonesia, yang menjadi fondasi utama bagi sistem rekomendasi yang akan dibangun. Secara keseluruhan, dataset ini berisi **1.025 baris unik** data, dengan **11 kolom** yang menyediakan detail lengkap dan bervariasi untuk setiap tempat wisata.

Berikut adalah uraian lengkap mengenai setiap variabel atau fitur yang terdapat pada dataset `wisata_indonesia_final.csv`:

  * **`kategori`**: Kolom ini menyimpan informasi mengenai jenis atau kategori utama dari tempat wisata, seperti 'air terjun', 'pantai', 'museum', 'gunung', 'taman', dan sebagainya. Fitur ini sangat penting untuk memahami preferensi pengguna dalam konteks jenis aktivitas atau daya tarik wisata.
  * **`nama_wisata`**: Merupakan nama spesifik dari objek wisata. Setelah dilakukan pemeriksaan awal, setiap `nama_wisata` dalam dataset ini teridentifikasi sebagai unik, artinya tidak ada duplikasi entri untuk tempat wisata yang sama. Ini memastikan bahwa setiap baris data merepresentasikan satu destinasi yang berbeda.
  * **`latitude`**: Menyimpan koordinat lintang geografis lokasi wisata. Fitur numerik ini krusial untuk informasi spasial, meskipun dalam proyek ini lebih difokuskan pada preferensi kategori dan provinsi.
  * **`longitude`**: Menyimpan koordinat bujur geografis lokasi wisata, melengkapi informasi spasial dari `latitude`.
  * **`alamat`**: Berisi alamat lengkap dari tempat wisata. Meskipun tidak digunakan secara langsung dalam perhitungan kemiripan teks, informasi ini penting untuk penyajian rekomendasi kepada pengguna.
  * **`provinsi`**: Menunjukkan provinsi tempat lokasi wisata berada. Pada pemeriksaan awal, ditemukan **90 nilai kosong (null)** pada kolom ini, yang memerlukan penanganan khusus dalam tahapan persiapan data. Fitur ini sangat penting karena pengguna dapat memfilter rekomendasi berdasarkan provinsi yang diinginkan.
  * **`kota_kabupaten`**: Menyimpan informasi kota atau kabupaten tempat lokasi wisata berada. Beberapa nilai kosong juga terdeteksi pada kolom ini, yang akan ditangani bersamaan dengan kolom `provinsi`.
  * **`nama_lengkap`**: Nama lengkap atau nama alternatif dari tempat wisata. Seringkali kolom ini menyerupai `nama_wisata` atau mengandung informasi alamat, sehingga relevansinya perlu dievaluasi lebih lanjut.
  * **`deskripsi`**: Kolom ini berisi deskripsi singkat atau informasi naratif tentang tempat wisata. Ini adalah fitur tekstual utama yang akan menjadi *input* penting untuk analisis teks menggunakan TF-IDF dalam sistem rekomendasi berbasis konten.
  * **`path`**: Menyimpan *path* relatif ke file data terkait. Kolom ini diidentifikasi tidak diperlukan untuk tujuan pemodelan rekomendasi dan akan dihapus.
  * **`path_gambar`**: Menyimpan *path* relatif ke file gambar terkait tempat wisata. Sama seperti kolom `path`, ini tidak digunakan secara langsung dalam perhitungan rekomendasi, namun bisa relevan untuk aplikasi *front-end*.

Dari analisis awal, dataset ini memiliki 1.025 entri unik dengan distribusi kategori dan provinsi yang bervariasi. **Kondisi data menunjukkan adanya *missing values* pada kolom `provinsi` dan `kota_kabupaten`**, serta potensi ketidaksesuaian atau *typo* pada kolom `kategori` dan karakter non-ASCII pada kolom teks. Semua isu ini akan menjadi fokus utama dalam tahapan Data Preparation untuk memastikan kualitas data yang optimal.

## Exploratory Data Analysis and Visualization

Exploratory Data Analysis (EDA) adalah langkah fundamental untuk memahami struktur, pola, dan anomali dalam data. Melalui visualisasi, kita dapat memperoleh wawasan yang lebih dalam mengenai karakteristik dataset wisata Indonesia.

1.  **Distribusi Jumlah Tempat Wisata per Provinsi**:
    Visualisasi ini berbentuk diagram batang horizontal yang menunjukkan sebaran objek wisata di setiap provinsi di Indonesia. Terlihat jelas bahwa **Jawa Timur** menjadi provinsi dengan jumlah objek wisata terbanyak (119 entri), diikuti oleh **Jawa Barat** (105 entri) dan **Jawa Tengah** (99 entri). Pola ini mengindikasikan bahwa Pulau Jawa memiliki konsentrasi destinasi wisata yang paling padat dalam dataset, yang mungkin mencerminkan kepadatan penduduk dan infrastruktur pariwisata yang lebih maju di pulau tersebut. Provinsi lainnya menunjukkan distribusi yang bervariasi, dengan beberapa provinsi di wilayah timur Indonesia memiliki jumlah objek wisata yang relatif sedikit.

2.  **Distribusi Kategori Tempat Wisata**:
    Diagram batang horizontal ini menampilkan 10 kategori tempat wisata teratas berdasarkan jumlah kemunculannya. Kategori **"wisata alam"** mendominasi dengan 169 objek, menunjukkan minat yang tinggi terhadap destinasi alam di Indonesia. Diikuti oleh **"pantai"** (104 objek) dan **"museum"** (90 objek). Kategori lain seperti gunung, mall, dan wisata kuliner juga memiliki jumlah yang signifikan. Visualisasi ini memberikan gambaran tentang preferensi umum jenis wisata yang ada dalam dataset, sekaligus menyoroti keragaman atraksi yang ditawarkan Indonesia.

3.  **Top 10 Kota/Kabupaten dengan Jumlah Wisata Terbanyak**:
    Visualisasi ini menampilkan 10 kota atau kabupaten yang memiliki jumlah objek wisata terbanyak. **Daerah Khusus Ibukota Jakarta** menempati posisi teratas dengan 73 objek wisata, menegaskan posisinya sebagai pusat aktivitas dan destinasi urban. Diikuti oleh **Badung** (30 objek) dan **Gianyar** (29 objek) di Bali, serta **Sleman** (22 objek) dan **Kota Bandung** (22 objek) di Jawa. Pola ini menunjukkan konsentrasi wisata yang kuat di ibu kota serta beberapa daerah populer di Bali dan Jawa, yang merupakan tujuan utama wisatawan.

4.  **Heatmap Jumlah Tempat Wisata per Provinsi dan Kategori (Top 10)**:
    Heatmap ini menyajikan matriks visual yang menunjukkan interaksi antara provinsi teratas dan kategori wisata teratas. Setiap sel berwarna menunjukkan jumlah tempat wisata untuk kombinasi provinsi-kategori tertentu.

      * **Jawa Timur**, **Jawa Barat**, dan **Jawa Tengah** menunjukkan konsentrasi tinggi pada berbagai kategori seperti "wisata alam", "pantai", dan "museum", menunjukkan kekayaan destinasi yang beragam di pulau Jawa.
      * **Bali** menonjol dengan jumlah "pantai" dan "wisata alam" yang signifikan, sejalan dengan citranya sebagai destinasi tropis yang terkenal.
      * **Daerah Istimewa Yogyakarta** menunjukkan dominasi pada kategori "museum" dan "candi", mencerminkan warisan budaya dan sejarah yang kuat.
        Heatmap ini secara visual menegaskan bahwa persebaran jenis wisata tidak merata di setiap provinsi, dan beberapa provinsi memiliki spesialisasi pada kategori tertentu, yang dapat menjadi wawasan penting untuk rekomendasi berbasis lokasi.

-----

## Data Preparation

Tahap persiapan data adalah fondasi krusial dalam setiap proyek *Machine Learning*. Tujuannya adalah untuk memastikan data bersih, konsisten, dan siap digunakan oleh model, sehingga dapat menghasilkan prediksi yang akurat dan dapat diandalkan. Proses ini dilakukan secara berurutan dan sistematis.

1.  **Penanganan *Missing Values***:

      * **Identifikasi Masalah**: Pemeriksaan awal menunjukkan bahwa kolom `provinsi` memiliki 90 nilai kosong (*null*), dan kolom `kota_kabupaten` juga mengandung beberapa nilai *null*. Keberadaan *missing values* ini dapat menyebabkan *error* atau bias pada model jika tidak ditangani.
      * **Solusi**: Untuk mengatasi masalah ini, nilai *null* pada kedua kolom (provinsi dan kota/kabupaten) diisi dengan nilai **modus** (nilai yang paling sering muncul) dari masing-masing kolom. Pendekatan ini dipilih karena `provinsi` dan `kota_kabupaten` adalah variabel kategorikal. Mengisi dengan modus adalah strategi yang efektif untuk mempertahankan distribusi data yang ada dan meminimalkan dampak pengisian pada karakteristik dataset secara keseluruhan. Setelah proses ini, semua *missing values* berhasil ditangani, memastikan kelengkapan data.

2.  **Penanganan Duplikasi Data**:

      * **Identifikasi Masalah**: Duplikasi baris data dapat mengganggu integritas model, menyebabkan *overfitting* atau bias karena model akan mempelajari pola yang sama berulang kali.
      * **Solusi**: Dilakukan pemeriksaan menyeluruh terhadap duplikasi baris data secara keseluruhan, serta duplikasi pada kolom `nama_wisata`. Hasil pemeriksaan menunjukkan bahwa **tidak ada duplikasi baris data yang ditemukan**, baik secara keseluruhan maupun pada kolom `nama_wisata`. Setiap entri dalam dataset unik dan merepresentasikan objek wisata yang berbeda. Ini adalah kondisi ideal yang mengurangi kebutuhan akan langkah penanganan duplikasi yang kompleks.

3.  **Pembersihan Teks (`deskripsi` dan `kota_kabupaten`)**:

      * **Identifikasi Masalah**: Kolom teks seperti `deskripsi` dan `kota_kabupaten` seringkali mengandung karakter non-ASCII (karakter yang tidak dikenali dalam standar teks umum), pola teks yang tidak relevan (misalnya format wiki), atau kalimat pembuka yang tidak informatif. Karakter rusak atau teks tidak relevan ini dapat mengganggu proses *vectorization* teks dan mengurangi kualitas rekomendasi.
      * **Solusi**:
          * **Fungsi `hapus_karakter_rusak`**: Dibuat untuk menghapus semua karakter non-ASCII dari string teks, menggantinya dengan spasi. Ini penting untuk memastikan teks bersih dan dapat diproses dengan benar oleh algoritma NLP.
          * **Fungsi `bersihkan_deskripsi`**: Dirancang khusus untuk membersihkan kolom `deskripsi` secara lebih mendalam. Fungsi ini melakukan beberapa langkah:
              * Menghapus pola teks yang umum ditemukan di artikel wiki (`== Judul ==` beserta isinya), yang sering kali tidak relevan untuk konteks rekomendasi.
              * Menghapus kalimat pembuka generik yang tidak memberikan informasi spesifik (misalnya, "Berikut ini adalah daftar...", "Latar belakang...", atau "Sejarah singkat...").
              * Menerapkan kembali `hapus_karakter_rusak` setelah pembersihan pola, untuk memastikan tidak ada karakter non-ASCII yang tersisa.
          * **Penerapan**: Fungsi `bersihkan_deskripsi` diterapkan pada kolom `deskripsi` untuk menghasilkan kolom baru bernama `deskripsi_bersih`. Kolom `kota_kabupaten` juga dibersihkan dari karakter non-ASCII. Proses ini memastikan bahwa teks yang akan digunakan untuk TF-IDF bersih, relevan, dan hanya mengandung informasi yang bermakna.

4.  **Standardisasi Kolom `kategori`**:

      * **Identifikasi Masalah**: Selama tahap EDA, teridentifikasi adanya *typo* pada kolom `kategori` (misalnya, 'wisate alam', 'wisath alam', 'wisatr alam', 'wisaub alam', 'wisawa alam', 'wisawi alam') yang seharusnya mengacu pada satu kategori yang sama: `wisata alam`. Ketidaksesuaian ini akan menyebabkan kategori yang sama dianggap berbeda oleh model, mengurangi akurasi analisis.
      * **Solusi**: Sebuah *mapping* (`kategori_mapping`) dibuat untuk mengganti semua *typo* tersebut menjadi `wisata alam`. Penerapan penggantian nilai ini memastikan konsistensi dan integritas data pada kolom `kategori`, yang sangat penting untuk klasifikasi dan analisis preferensi pengguna.

5.  **Penghapusan Kolom Tidak Diperlukan**:

      * **Alasan**: Untuk mengurangi dimensi data dan fokus pada fitur-fitur yang esensial untuk model rekomendasi, kolom `path` dan kolom `deskripsi` yang asli dihapus dari *DataFrame*. Kolom `path` tidak relevan untuk pemodelan rekomendasi, sedangkan informasi dari kolom `deskripsi` telah dibersihkan dan dipindahkan ke kolom baru `deskripsi_bersih` yang akan digunakan sebagai *input* utama dalam model.

-----

## Modeling

Tahapan ini adalah inti dari proyek, di mana model sistem rekomendasi dibangun dan dilatih untuk menyelesaikan permasalahan personalisasi. Model ini dirancang sebagai sistem hibrida yang mengoptimalkan kekuatan dari *content-based filtering* dan *neural network*.

### Pendekatan Solusi: Hybrid Recommendation System

Sistem rekomendasi yang dikembangkan mengadopsi pendekatan hibrida, menggabungkan dua strategi utama untuk menghasilkan skor relevansi yang komprehensif:

1.  **Content-Based Filtering dengan TF-IDF Cosine Similarity (Weight 0.7)**:

      * **Kelebihan**: Metode ini sangat efektif dalam menangkap relevansi semantik berdasarkan konten tekstual deskripsi tempat wisata. Keunggulannya terletak pada transparansi (mudah menjelaskan mengapa rekomendasi diberikan berdasarkan kesamaan fitur) dan kemampuannya menangani *cold-start problem* untuk pengguna baru, karena tidak memerlukan data interaksi pengguna lain. Cepat dalam perhitungan dan adaptif terhadap perubahan profil preferensi.
      * **Kekurangan**: Rentan terhadap *over-specialization* (cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai, membatasi eksplorasi). Kualitas rekomendasi sangat bergantung pada kekayaan dan kedalaman deskripsi teks item. Selain itu, ia mungkin kurang mampu menangkap pola atau hubungan kontekstual yang lebih kompleks di luar kesamaan kata kunci langsung.

2.  **Model Neural Network (Weight 0.3)**:

      * **Kelebihan**: Model *neural network* (khususnya *Multi-Layer Perceptron* yang digunakan di sini) sangat kuat dalam mempelajari hubungan non-linear dan pola kompleks antar fitur yang mungkin tidak dapat ditangkap oleh metode linier sederhana. Model ini fleksibel untuk mengintegrasikan berbagai jenis fitur (kategorikal, numerik, biner) dan memiliki potensi kinerja yang sangat tinggi jika dilatih dengan data yang memadai.
      * **Kekurangan**: Sering disebut sebagai "kotak hitam" karena sulit untuk menginterpretasikan bagaimana model mencapai keputusan tertentu, mengurangi transparansi. Membutuhkan data pelatihan yang cukup besar dan berkualitas tinggi, serta sumber daya komputasi yang signifikan dan *hyperparameter tuning* yang cermat.

**Strategi Pembobotan Hybrid**:
Kedua skor dari metode di atas digabungkan dengan **pembobotan**: `cosine_score` (dari TF-IDF) diberi bobot **0.7**, sedangkan `model_score` (dari *neural network*) diberi bobot **0.3**. Kombinasi ini menghasilkan `final_score`. Pembobotan ini dipilih untuk memberikan prioritas lebih pada kemiripan konten langsung (yang seringkali merupakan indikator kuat preferensi eksplisit pengguna), sementara model *neural network* menambahkan lapisan kecerdasan untuk menangkap pola yang lebih kompleks dan interaksi fitur terstruktur yang mungkin tidak langsung terlihat dari teks saja.

### Tahapan Pemodelan

1.  **Penyaringan Data Berdasarkan Preferensi Pengguna**:
    Langkah pertama adalah menyaring dataset asli (`df_clean`) hanya untuk tempat wisata yang berada di **provinsi yang dipilih oleh pengguna** (misalnya, "Jawa Timur"). Hal ini memastikan bahwa rekomendasi yang diberikan relevan secara geografis. Sebuah kolom `label` juga dibuat sebagai *ground truth* untuk tujuan evaluasi, dengan nilai 1 jika kategori wisata termasuk dalam `preferred_categories` pengguna, dan 0 jika tidak.

2.  **TF-IDF Vectorization dan Perhitungan Cosine Similarity**:

      * `TfidfVectorizer` diinisialisasi dengan *stopwords* bahasa Indonesia untuk mengubah teks `deskripsi_bersih` dari setiap tempat wisata menjadi representasi numerik (matriks TF-IDF). Ini membantu dalam menyoroti kata-kata penting yang membedakan satu deskripsi dari yang lain.
      * Preferensi kategori pengguna (misalnya, "gunung taman pantai") digabungkan menjadi satu string dan juga diubah menjadi vektor TF-IDF.
      * **`cosine_similarity`** dihitung antara vektor preferensi pengguna dan setiap baris dalam matriks TF-IDF dari deskripsi wisata. Hasilnya adalah `cosine_score`, yang mengukur kemiripan tekstual antara preferensi pengguna dan deskripsi setiap tempat wisata. Skor ini menjadi salah satu komponen utama dalam `final_score` dan juga berfungsi sebagai target pelatihan untuk model *neural network*.

3.  **Feature Engineering untuk Model Neural Network**:
    Untuk melatih model *neural network*, beberapa fitur baru direkayasa dari data yang sudah ada:

      * **`desc_len`**: Menghitung jumlah kata dalam kolom `deskripsi_bersih`. Fitur ini dapat memberikan informasi tentang seberapa detail atau kompleks deskripsi suatu tempat wisata.
      * **`kategori_enc`**: Kolom kategorikal `kategori` diubah menjadi representasi numerik menggunakan `LabelEncoder`. Ini memungkinkan model *neural network* untuk memproses kategori sebagai fitur numerik.
      * **`provinsi_enc`**: Serupa dengan `kategori_enc`, kolom `provinsi` juga diubah menjadi representasi numerik menggunakan `LabelEncoder`.
      * **`is_preferred_category`**: Sebuah fitur biner (0 atau 1) yang secara eksplisit menandai apakah kategori wisata tersebut termasuk dalam daftar kategori yang disukai oleh pengguna. Fitur ini membantu model secara langsung mengidentifikasi kesesuaian kategori.

4.  **Normalisasi Fitur dan Pembagian Data**:

      * Fitur-fitur yang akan menjadi *input* untuk model *neural network* (`kategori_enc`, `provinsi_enc`, `desc_len`, `is_preferred_category`) dinormalisasi menggunakan `MinMaxScaler`. Normalisasi ini menskalakan semua fitur ke dalam rentang 0 hingga 1. Hal ini krusial karena *neural network* cenderung berkinerja lebih baik ketika *input* berada dalam skala yang sama, membantu dalam proses konvergensi dan mencegah fitur dengan skala besar mendominasi pembelajaran.
      * Data yang telah dinormalisasi (X) dan target (`cosine_score` sebagai y) kemudian dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan `train_test_split` dengan `random_state` yang tetap. Pembagian ini memastikan bahwa model dievaluasi pada data yang belum pernah dilihat sebelumnya, memberikan gambaran yang objektif tentang kinerja model.

5.  **Pembangunan dan Pelatihan Model Neural Network (Keras)**:

      * Model *neural network* dibangun menggunakan Keras `Sequential` API, yang memungkinkan pembuatan model lapisan demi lapisan. Arsitektur model terdiri dari:
          * **Lapisan Input**: Sesuai dengan jumlah fitur yang telah dinormalisasi.
          * **Lapisan Tersembunyi (Dense)**: Dua lapisan `Dense` dengan 64 dan 32 neuron, masing-masing menggunakan fungsi aktivasi ReLU (`Rectified Linear Unit`) yang populer karena kemampuannya dalam menangani non-linearitas dan mencegah *vanishing gradient*.
          * **Lapisan Dropout**: Setelah setiap lapisan `Dense`, ditambahkan lapisan `Dropout` (dengan *rate* 0.3 dan 0.2). *Dropout* adalah teknik regularisasi yang secara acak menonaktifkan sebagian neuron selama pelatihan, efektif dalam mencegah *overfitting* dan meningkatkan generalisasi model.
          * **Lapisan Output**: Satu neuron dengan fungsi aktivasi Sigmoid. Fungsi Sigmoid menghasilkan *output* antara 0 dan 1, yang ideal untuk memprediksi skor relevansi yang juga berada dalam rentang tersebut.
      * **Kompilasi Model**: Model dikompilasi dengan:
          * *Optimizer* **`Adam`** (dengan *learning rate* 0.001): Sebuah algoritma optimasi adaptif yang efisien untuk pelatihan *deep learning*.
          * Fungsi *loss* **`mse` (Mean Squared Error)**: Metrik ini digunakan untuk mengukur selisih rata-rata kuadrat antara nilai prediksi model dan nilai `cosine_score` sebenarnya.
          * Metrik **`mae` (Mean Absolute Error)**: Digunakan untuk memantau rata-rata nilai absolut dari *error* selama pelatihan.
      * **Pelatihan Model**: Model dilatih selama 30 *epoch* dengan *batch size* 16. Selama pelatihan, *loss* dan *MAE* pada data pelatihan maupun validasi menunjukkan penurunan yang konsisten dan signifikan, menandakan bahwa model belajar dengan efektif dan berhasil memetakan fitur *input* ke skor kemiripan secara akurat tanpa menunjukkan tanda-tanda *overfitting* yang parah.

### Rekomendasi dari Preferensi Pengguna

Fungsi `rekomendasi_wisata_berbobot` mengintegrasikan seluruh tahapan pemodelan di atas untuk menghasilkan daftar rekomendasi akhir. Fungsi ini secara dinamis:

1.  Memfilter data wisata hanya untuk provinsi yang dipilih pengguna.
2.  Menghitung `cosine_score` (kemiripan TF-IDF) antara preferensi kategori pengguna dan deskripsi wisata yang difilter.
3.  Melakukan *feature engineering* dan normalisasi pada fitur-fitur yang akan menjadi *input* model *neural network*.
4.  Memprediksi `model_score` menggunakan model *neural network* yang sudah dilatih.
5.  Menggabungkan `cosine_score` dan `model_score` dengan bobot yang telah ditentukan (0.7 dan 0.3) untuk menghasilkan `final_score`.
6.  Mengurutkan tempat wisata berdasarkan `final_score` dari tertinggi ke terendah dan mengambil 10 teratas sebagai rekomendasi terbaik. Output yang disajikan mencakup `nama_wisata`, `provinsi`, `alamat`, `deskripsi_bersih`, `kota_kabupaten`, serta detail `final_score`, `cosine_score`, dan `model_score` untuk setiap rekomendasi.

-----

## Evaluation

Evaluasi model adalah tahapan krusial untuk mengukur seberapa baik sistem rekomendasi dalam memberikan rekomendasi yang relevan dan akurat kepada pengguna. Metrik yang digunakan dipilih secara hati-hati agar sesuai dengan konteks permasalahan dan tujuan bisnis.

### Metrik Evaluasi

1.  **Precision@K**:
    Precision@K adalah metrik yang sangat relevan dalam sistem rekomendasi karena berfokus pada kualitas rekomendasi teratas yang disajikan kepada pengguna. Metrik ini mengukur proporsi item yang direkomendasikan yang benar-benar relevan di antara *K* item teratas yang diberikan oleh sistem.
    $$\text{Precision@K} = \frac{\text{Jumlah item relevan di top K}}{\text{K}}$$
    Dalam konteks proyek ini, "item relevan" didefinisikan sebagai tempat wisata yang kategorinya termasuk dalam `preferred_categories` yang telah ditentukan oleh pengguna. Nilai Precision@K yang tinggi menunjukkan bahwa sistem sangat tepat dalam memberikan rekomendasi yang sesuai dengan minat utama pengguna.

2.  **Recall@K**:
    Recall@K, di sisi lain, mengukur cakupan sistem rekomendasi. Metrik ini menunjukkan proporsi item relevan yang berhasil direkomendasikan di antara *K* item teratas, dibandingkan dengan total seluruh item relevan yang sebenarnya ada dalam dataset.
    $$\text{Recall@K} = \frac{\text{Jumlah item relevan di top K}}{\text{Total jumlah item relevan}}$$
    Recall@K sangat penting untuk memahami seberapa lengkap sistem dalam menangkap semua item relevan yang mungkin diminati pengguna, meskipun tidak harus berada di posisi teratas.

3.  **Mean Squared Error (MSE)** dan **Mean Absolute Error (MAE)**:
    Kedua metrik ini digunakan secara khusus untuk mengevaluasi kinerja model *neural network* dalam memprediksi skor relevansi.

      * **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat dari selisih antara nilai prediksi model dan nilai sebenarnya (`cosine_score`). MSE memberikan bobot yang lebih besar pada kesalahan prediksi yang besar, membuatnya sensitif terhadap *outlier*. Formula MSE adalah:
        $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
        di mana $y\_i$ adalah nilai sebenarnya, $\\hat{y}\_i$ adalah nilai prediksi, dan $n$ adalah jumlah sampel.
      * **Mean Absolute Error (MAE)**: Mengukur rata-rata nilai absolut dari selisih antara nilai prediksi model dan nilai sebenarnya. MAE memberikan ukuran rata-rata seberapa jauh prediksi model dari nilai sebenarnya, dan cenderung lebih tahan terhadap *outlier* dibandingkan MSE. Formula MAE adalah:
        $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
        Nilai MSE dan MAE yang rendah mengindikasikan bahwa model *neural network* membuat prediksi yang sangat dekat dengan skor relevansi yang sebenarnya.

### Hasil Evaluasi

1.  **Precision@10 dan Recall@10 (untuk Rekomendasi)**:

      * **Precision@10: 1.00**
        Hasil ini sangatlah mengesankan dan menunjukkan kinerja yang luar biasa dalam hal ketepatan. Artinya, dari 10 rekomendasi teratas yang diberikan oleh sistem, **semua (100%)** adalah tempat wisata yang memang termasuk dalam kategori preferensi pengguna (yaitu "gunung", "taman", atau "pantai"). Ini menegaskan bahwa sistem sangat akurat dan dapat diandalkan dalam menyajikan rekomendasi yang benar-benar sesuai dengan minat utama pengguna. Tidak ada "sampah" atau rekomendasi yang tidak relevan di antara yang teratas.
      * **Recall@10: 0.28**
        Nilai Recall@10 sebesar 0.28 berarti bahwa sistem berhasil merekomendasikan sekitar **28%** dari total seluruh objek wisata yang sebenarnya relevan dengan kategori favorit pengguna yang ada di provinsi Jawa Timur, dalam daftar 10 rekomendasi teratas. Meskipun Precision sangat tinggi, Recall yang relatif lebih rendah menunjukkan bahwa meskipun rekomendasi yang diberikan sangat tepat, masih ada banyak tempat wisata relevan lainnya yang tidak masuk dalam daftar 10 rekomendasi teratas. Ini bisa mengindikasikan bahwa sistem cenderung fokus pada rekomendasi yang paling kuat dan jelas kesesuaiannya, dan mungkin melewatkan beberapa opsi relevan lainnya yang memiliki skor relevansi yang lebih rendah atau kurang menonjol.

2.  **Loss dan MAE Model Neural Network pada Data Uji**:

      * **Loss (MSE): 0.0046**
      * **MAE: 0.0394**
        Nilai *loss* (MSE) dan *MAE* yang sangat rendah pada data uji mengindikasikan bahwa model *neural network* mampu memprediksi skor relevansi dengan tingkat kesalahan yang minimal dan akurasi yang tinggi. Angka-angka ini menunjukkan bahwa model terlatih dengan sangat baik, memiliki kemampuan generalisasi yang kuat terhadap data yang belum pernah dilihat sebelumnya, dan prediksinya sangat dekat dengan nilai *cosine similarity* sebenarnya. Ini adalah indikator kuat bahwa model *neural network* berkontribusi secara efektif dalam memprediksi relevansi.

### Visualisasi Performa Model (Training History)

Visualisasi ini membantu dalam memahami bagaimana model *neural network* belajar dan beradaptasi selama proses pelatihan.

  * **Model Loss during Training**:
    Grafik ini menampilkan nilai *loss* (Mean Squared Error) untuk data pelatihan dan data validasi sepanjang *epoch* pelatihan. Terlihat bahwa baik *training loss* maupun *validation loss* menunjukkan penurunan yang signifikan pada *epoch-epoch* awal (sekitar 1-10 *epoch*), kemudian melambat dan stabil pada nilai yang sangat rendah setelah sekitar 15 *epoch*. Kurva *validation loss* yang bergerak paralel dan relatif dekat dengan kurva *training loss* mengindikasikan bahwa model belajar dengan baik dan **tidak menunjukkan tanda-tanda *overfitting* yang parah**. Ini berarti model mampu menggeneralisasi dengan baik pada data yang tidak terlihat selama pelatihan.

  * **Model MAE during Training**:
    Grafik ini menunjukkan nilai Mean Absolute Error (MAE) untuk data pelatihan dan data validasi sepanjang *epoch* pelatihan. Serupa dengan grafik *loss*, *training MAE* dan *validation MAE* juga menunjukkan penurunan yang stabil dan signifikan pada *epoch* awal, kemudian stabil pada nilai yang rendah. Penurunan MAE yang konsisten dan nilai yang sangat rendah di akhir pelatihan menegaskan bahwa model mampu membuat prediksi dengan kesalahan rata-rata yang sangat kecil. Kesenjangan yang kecil antara *training MAE* dan *validation MAE* lebih lanjut mengonfirmasi bahwa model memiliki kinerja yang konsisten dan baik pada data yang tidak terlihat, tanpa tanda-tanda *overfitting* yang jelas.

Secara keseluruhan, hasil evaluasi menunjukkan bahwa sistem rekomendasi hibrida ini memiliki performa yang sangat kuat. **Precision@10 yang sempurna** adalah bukti bahwa rekomendasi teratas yang diberikan sangat akurat dan relevan dengan preferensi pengguna. Meskipun Recall masih memiliki ruang untuk peningkatan cakupan, sistem ini berhasil memenuhi tujuan utama yaitu memberikan rekomendasi yang presisi dan relevan. Kinerja model *neural network* yang solid, ditunjukkan oleh nilai MSE dan MAE yang rendah, serta stabilitas pada grafik *training history*, menegaskan fondasi yang kuat untuk sistem rekomendasi ini.

-----
