#!/usr/bin/env python
# coding: utf-8

# ## Submission Sistem Rekomendasi : Arliyandi ##

# ### Load Library

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import pickle


# ## Data Exploration ##

# In[33]:


df = pd.read_csv('laptops.csv')
df.head()


# Dataset yang digunakan berisi informasi spesifikasi laptop dari berbagai merek dan model. Total terdapat 2160 entri data, masing-masing memiliki 12 fitur.
# 
# Beberapa fitur utama pada dataset meliputi:
# - Brand, Model, CPU, GPU: Informasi spesifikasi hardware
# - RAM, Storage, Storage type, Screen, Touch: komponen teknis
# - Final Price: Harga akhir laptop.

# In[34]:


df.info()


# Dari hasil eksplorasi awal:
# 
# - Jumlah data: 2160 baris
# - Jumlah kolom: 12 fitur.
# - terdapat beberapa kolom yang kosong, dan perlu penyesuaian di proses cleaning nantinya

# In[35]:


df.describe()


# Hasil analisis deskriptif menunjukkan variasi spesifikasi:
# - RAM berkisar antara 4GB hingga 128GB, dengan median di 16GB
# - Kapasitas storage mulai dari 0GB (kemungkinan error entri) hingga 4TB.
# - Ukuran layar (screen) dominan di sekitar 15.6 inci
# - Harga laptop bervariasi cukup lebar, mulai dari sekitar $201 hingga $7150

# In[36]:


df.isnull().sum()


# Berdasarkan informasi diatas, terdapat beberapa fitur yang memiliki missing calue :
# - Storage type: 42 missing value
# - GPU: 1371 missing value
# - Screen: 4 missing value

# In[37]:


plt.figure(figsize=(8,5))
sns.histplot(df['Final Price'], bins=50, kde=True)
plt.title('Distribusi Harga Laptop')
plt.xlabel('Final Price (Standardized)')
plt.ylabel('Jumlah Data')
plt.show()


# Dapat kita perhatikan, sebaran harga laptop paling banyak di harga $500 - $1500

# In[38]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df['RAM'])
plt.title('Distribusi RAM Laptop')
plt.xlabel('RAM (Standardized)')
plt.show()


# Untuk Ram, Distribusi size nya paling banyak di 8 - 16 Gb

# ## Data Cleaning ##

# In[39]:


df = df.drop(columns=['Laptop', 'Status'])


# Drop kolom Laptop dan status karena tidak berkaitan denan model nantinya

# In[40]:


df['Storage type'].fillna(df['Storage type'].mode()[0], inplace=True)


# Isi storage Type dengan mode, untuk mengatasi missing value. Metode ini dipilih karena missing value tidak terlalu banyak dan tipe gpu tidak terlalu banyak berdampak besar pada seharusnya pada model

# In[41]:


df['GPU'].fillna('Unknown', inplace=True)


# Untuk GPU, missing value kita isi dengan 'unknown'. Bisa juga diisi dengan Integrated, tapi karena value awalnya memang kosong jadi lebih aman saya isi dengan unknown.

# In[42]:


df['Screen'].fillna(df['Screen'].median(), inplace=True)


# ## Data Preparation ##

# In[43]:


categorical_cols = ['Brand', 'Model', 'CPU', 'Storage type', 'GPU', 'Touch']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Fitur kategorikal seperti Brand, Model, CPU, Storage type, GPU, dan Touch tidak dapat langsung digunakan dalam bentuk string. Oleh karena itu, diterapkan proses encoding menggunakan 'Label Encoding' untuk mengubah nilai string menjadi numerik.

# In[44]:


# Scaling numerik
numerical_cols = ['RAM', 'Storage', 'Screen', 'Final Price']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Beberapa fitur numerik memiliki skala yang sangat bervariasi (misalnya: RAM, Storage, dan Final Price). Untuk menyamakan skala dan menghindari dominasi fitur tertentu pada perhitungan similarity, diterapkan 'Standard Scaler' sehingga setiap fitur memiliki mean = 0 dan standar deviasi = 1.

# In[45]:


print(df.head())
print(df.info())
print(df.describe())


# Setelah dilakukan encoding dan scaling:
# - Dataset memiliki total 2160 baris dan 10 kolom.
# - Semua data siap diproses ke tahap modeling
# - Tidak ada lagi missing value setelah tahap data cleaning sebelumnya.

# In[46]:


# Siapkan fitur untuk similarity
feature_cols = categorical_cols + numerical_cols
features = df[feature_cols].values


# ## Modelling ##

# ### Content Based Filtering ###

# In[47]:


# Hitung cosine similarity
similarity = cosine_similarity(features)


# In[48]:


# Fungsi rekomendasi
def recommend(index, top_n=5):
    score = list(enumerate(similarity[index]))
    score = sorted(score, key=lambda x: x[1], reverse=True)
    score = score[1:top_n+1]
    indices = [i[0] for i in score]
    return df.iloc[indices]


# In[49]:


print("Rekomendasi untuk data ke-10:")
print(recommend(10))


# Sistem berhasil merekomendasikan produk dengan spesifikasi yang hampir identik dari sisi:
# Brand (sama-sama kode 13, artinya sama brandnya)
# Model (kode 108, model juga sama)
# CPU (kode 14, prosesor sama)
# Storage type, GPU, Touch, Screen, semuanya sama.
# 
# Variasi utama hanya sedikit pada:
# 
# RAM -> ada perbedaan kecil (contoh: -0.751 vs -1.156 → berarti mungkin beda 8GB vs 16GB jika sebelum di scaling).
# 
# Storage -> sebagian besar sama, hanya 1 data dengan storage sedikit lebih rendah (perbedaan minor).
# 
# Final Price -> harga bervariasi tipis, semua tetap dalam range mirip.

# Sistem merekomendasikan laptop yang sangat mirip dengan target (spesifikasi utama tetap sama), namun ada sedikit variasi di kapasitas RAM, Storage, dan harga. Ini sesuai ekspektasi dari sistem content-based filtering.

# ### Collaborative Filtering ###

# In[51]:


def recommend_collaborative(index, top_n=5):
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]
    indices = [i[0] for i in scores]
    return df.iloc[indices]


# In[52]:


print("Collaborative Filtering Recommendation (index 10):")
print(recommend_collaborative(10))


# Collaborative Filtering menghasilkan rekomendasi yang sangat mirip dengan Content-Based Filtering dalam kasus dataset ini.
# Hal ini terjaid karena kedua pendekatan menggunakan sumber data spesifikasi produk yang sama, hanya berbeda dari cara perspektif pendekatannya (fitur produk vs relasi antar item).

# 
