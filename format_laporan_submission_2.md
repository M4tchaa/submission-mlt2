# Laporan Proyek Machine Learning - Arliyandi

## Project Overview

Laptop sudah hampir menjadi kebutuhan primer bagi masyarakat saat ini. Pembelian perangkat laptop sering kali dihadapkan pada banyak pilihan spesifikasi, harga, dan brand. Calon pembeli kerap kesulitan dalam memilih laptop yang paling sesuai dengan kebutuhan spesifik maupun anggaran yang dimiliki. Oleh karena itu, pengembangan sistem rekomendasi berbasis spesifikasi produk dapat membantu mempermudah proses pengambilan keputusan pembelian laptop.

Source :
- Hill, N., & Alexander, J. (2017). The handbook of customer satisfaction and loyalty measurement (3rd Edition). Routledge.
- Loeffler, J. (2024). Chromebooks vs Laptops: which is best for students? Techradar.Com.
- T. Ricci, L. Rokach, B. Shapira, *Recommender Systems Handbook*, Springer, 2015.

## Business Understanding

### Problem Statements

- Banyaknya variasi spesifikasi laptop menyebabkan calon pembeli kesulitan memilih produk yang tepat.
- Tidak tersedianya sistem rekomendasi yang memanfaatkan kesamaan fitur teknis antar laptop.

### Goals

- Membantu calon pembeli menemukan laptop yang relevan dengan spesifikasi yang mereka inginkan.
- Mengembangkan sistem rekomendasi berbasis kesamaan spesifikasi produk.

### Solution statements
- Menggunakan pendekatan **Content-Based Filtering**, Membandingkan kemiripan spesifikasi laptop secara langsung menggunakan cosine similarity antar vektor fitur produk.
- Menggunakan pendekatana **Collaborative Filtering**, Menghitung kemiripan antar item produk, yang dalam kasus ini disimulasikan menggunakan kesamaan fitur antar produk (karena tidak tersedia data interaksi user).

## Data Understanding
Dataset yang digunakan berasal dari https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset . Dataset berisi 2160 baris data laptop, dengan 12 fitur utama:

- **Laptop**: Nama produk (tidak digunakan dalam modeling).
- **Status**: Status produk (New) - tidak digunakan.
- **Brand**: Merek laptop.
- **Model**: Model produk.
- **CPU**: Tipe prosesor.
- **RAM**: Kapasitas RAM (GB).
- **Storage**: Kapasitas penyimpanan (GB).
- **Storage type**: Jenis penyimpanan (SSD/HDD).
- **GPU**: Jenis kartu grafis.
- **Screen**: Ukuran layar (inch).
- **Touch**: Layar sentuh (Yes/No).
- **Final Price**: Harga akhir (USD).

### Kondisi Data:

- Missing value:
  - Storage type: 42 data
  - GPU: 1371 data
  - Screen: 4 data
- Harga laptop berkisar dari $201 hingga $7150.
- RAM bervariasi antara 4GB hingga 128GB.
- Ukuran layar didominasi pada 15.6 inch.

### Visualisasi (EDA)

Visualisasi sederhana dilakukan untuk melihat sebaran data:

- **Distribusi Harga:** mayoritas produk berada pada harga menengah, terdapat outlier harga sangat tinggi.
- **Distribusi RAM:** sebagian besar laptop berada pada rentang RAM standar (8GB - 16GB), dengan sedikit outlier di kapasitas RAM yang tinggi.

## Data Preparation
Beberapa tahapan preparation yang dilakukan:

1. **Handling Missing Value**
   - Storage type -> diisi dengan mode.
   - GPU -> diisi dengan string 'Unknown'
   - Screen -> diisi dengan median.

2. **Encoding Kategorikal**
   - Kolom: `Brand`, `Model`, `CPU`, `Storage type`, `GPU`, dan `Touch` diubah menggunakan Label Encoding

3. **Scaling Numerik**
   - Kolom numerik `RAM`, `Storage`, `Screen`, dan `Final Price` distandarisasi menggunakan StandardScaler untuk menyamakan skala perhitungan similarity.


## Modeling

### Content-Based Filtering
Metode Content-Based Filtering membandingkan kesamaan fitur antar produk menggunakan cosine similarity.

#### Insight menggunakan Content-Based Filtering:

- Sistem berhasil merekomendasikan produk dengan spesifikasi hampir identik pada Brand, Model, CPU, Storage type, GPU, Touch, dan Screen.
- Variasi kecil ditemukan pada RAM (contoh: perbedaan -0.751 vs -1.156 menunjukkan kemungkinan 8GB vs 16GB sebelum scaling), serta Storage minor.
- Harga bervariasi tipis, tetap dalam range harga serupa.

### Collaborative Filtering (Item-Based)
Collaborative Filtering disimulasikan dengan pendekatan item-based similarity antar produk menggunakan fitur spesifikasi.

#### Insight Collaborative Filtering:

- Rekomendasi collaborative menghasilkan produk-produk dengan spesifikasi hampir identik seperti content-based.
- Sistem mampu mengidentifikasi produk serupa dari sisi item-item meskipun tanpa data interaksi pengguna.
- Variasi minor tetap terjadi pada RAM, Storage, dan harga, serupa dengan content-based.

## Evaluation
Karena tidak terdapat data interaksi adri user (seperti rating, feedback, dsb), evaluasi dilakukan dengan observasi dari hasil rekomendasi itu sendiri.

Berikut adalah hasil evaluasinya :
- Sistem berhasil mengembalikan produk dengan kemiripan spesifikasi tinggi.
- Model sudah mampu menangani variasi minor antar fitur.
- Metrik evaluasi numerik (seperti precision, recall, RMSE) tidak digunakan karena proyek ini murni berbasis similarity dari data, bukan supervised learning.


**---Ini adalah bagian akhir laporan---**
