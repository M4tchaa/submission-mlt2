#!/usr/bin/env python
# coding: utf-8

# ## Submission Sistem Rekomendasi : Arliyandi ##

# ### Load Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import pickle


# ## Data Exploration ##

# In[2]:


df = pd.read_csv('laptops.csv')
df.head()


# Dataset yang digunakan berisi informasi spesifikasi laptop dari berbagai merek dan model. Total terdapat 2160 entri data, masing-masing memiliki 12 fitur.
# 
# Beberapa fitur utama pada dataset meliputi:
# - Brand, Model, CPU, GPU: Informasi spesifikasi hardware
# - RAM, Storage, Storage type, Screen, Touch: komponen teknis
# - Final Price: Harga akhir laptop.

# In[3]:


df.info()


# Dari hasil eksplorasi awal:
# 
# - Jumlah data: 2160 baris
# - Jumlah kolom: 12 fitur.
# - terdapat beberapa kolom yang kosong, dan perlu penyesuaian di proses cleaning nantinya

# In[4]:


df.describe()


# Hasil analisis deskriptif menunjukkan variasi spesifikasi:
# - RAM berkisar antara 4GB hingga 128GB, dengan median di 16GB
# - Kapasitas storage mulai dari 0GB (kemungkinan error entri) hingga 4TB.
# - Ukuran layar (screen) dominan di sekitar 15.6 inci
# - Harga laptop bervariasi cukup lebar, mulai dari sekitar $201 hingga $7150

# In[5]:


df.isnull().sum()


# Berdasarkan informasi diatas, terdapat beberapa fitur yang memiliki missing calue :
# - Storage type: 42 missing value
# - GPU: 1371 missing value
# - Screen: 4 missing value

# In[6]:


plt.figure(figsize=(8,5))
sns.histplot(df['Final Price'], bins=50, kde=True)
plt.title('Distribusi Harga Laptop')
plt.xlabel('Final Price (Standardized)')
plt.ylabel('Jumlah Data')
plt.show()


# Dapat kita perhatikan, sebaran harga laptop paling banyak di harga $500 - $1500

# In[7]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df['RAM'])
plt.title('Distribusi RAM Laptop')
plt.xlabel('RAM (Standardized)')
plt.show()


# Untuk Ram, Distribusi size nya paling banyak di 8 - 16 Gb

# ## Data Cleaning ##

# In[8]:


df = df.drop(columns=['Laptop', 'Status'])


# Drop kolom Laptop dan status karena tidak berkaitan denan model nantinya

# In[9]:


df['Storage type'].fillna(df['Storage type'].mode()[0], inplace=True)


# Isi storage Type dengan mode, untuk mengatasi missing value. Metode ini dipilih karena missing value tidak terlalu banyak dan tipe gpu tidak terlalu banyak berdampak besar pada seharusnya pada model

# In[10]:


df['GPU'].fillna('Unknown', inplace=True)


# Untuk GPU, missing value kita isi dengan 'unknown'. Bisa juga diisi dengan Integrated, tapi karena value awalnya memang kosong jadi lebih aman saya isi dengan unknown.

# In[11]:


df['Screen'].fillna(df['Screen'].median(), inplace=True)


# ## Data Preparation ##

# In[12]:


categorical_cols = ['Brand', 'Model', 'CPU', 'Storage type', 'GPU', 'Touch']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Fitur kategorikal seperti Brand, Model, CPU, Storage type, GPU, dan Touch tidak dapat langsung digunakan dalam bentuk string. Oleh karena itu, diterapkan proses encoding menggunakan 'Label Encoding' untuk mengubah nilai string menjadi numerik.

# In[13]:


# Scaling numerik
numerical_cols = ['RAM', 'Storage', 'Screen', 'Final Price']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Beberapa fitur numerik memiliki skala yang sangat bervariasi (misalnya: RAM, Storage, dan Final Price). Untuk menyamakan skala dan menghindari dominasi fitur tertentu pada perhitungan similarity, diterapkan 'Standard Scaler' sehingga setiap fitur memiliki mean = 0 dan standar deviasi = 1.

# In[14]:


print(df.head())
print(df.info())
print(df.describe())


# Setelah dilakukan encoding dan scaling:
# - Dataset memiliki total 2160 baris dan 10 kolom.
# - Semua data siap diproses ke tahap modeling
# - Tidak ada lagi missing value setelah tahap data cleaning sebelumnya.

# In[15]:


# Siapkan fitur untuk similarity
feature_cols = categorical_cols + numerical_cols
features = df[feature_cols].values


# ## Modelling ##

# ### Content Based Filtering ###

# In[16]:


# Hitung cosine similarity
similarity = cosine_similarity(features)


# In[17]:


# Fungsi rekomendasi
def recommend(index, top_n=5):
    score = list(enumerate(similarity[index]))
    score = sorted(score, key=lambda x: x[1], reverse=True)
    score = score[1:top_n+1]
    indices = [i[0] for i in score]
    return df.iloc[indices]


# In[18]:


print("Top-5 rekomendasi untuk data ke-10 (Content-Based Filtering):")
recommend(10)


# In[ ]:


def evaluate_mean_similarity(index, top_n=5):
    score = list(enumerate(similarity[index]))
    score = sorted(score, key=lambda x: x[1], reverse=True)
    score = score[1:top_n+1] #di +1 karena dikecualikan index yg dijadikan percobaan
    similarities = [x[1] for x in score]
    return sum(similarities) / top_n


# In[21]:


score_mean_cosine = evaluate_mean_similarity(10, top_n=5)
print(f"Mean Cosine Similarity untuk index yg dipilih: {score_mean_cosine:.4f}")


# Semua rekomendasi memiliki kesamaan penuh (cosine similarity = 1) terhadap item acuan — artinya vektor fitur mereka identik setelah proses normalisasi.
# 
# Ini terjadi kemungkinan karena fitur utama seperti Brand, Model, CPU, GPU, Storage type, dan Screen identik, dan hanya RAM, Storage, atau Final Price yang sedikit berbeda — yang tidak cukup besar untuk menurunkan similarity.

# Berdasarkan perhitungan Mean Cosine Similarity, sistem menghasilkan skor 1.0000 untuk data ke-10. Ini menunjukkan bahwa seluruh item yang direkomendasikan sangat mirip secara fitur dengan item tersebut. Rekomendasi ini relevan untuk pendekatan Content-Based Filtering yang mengandalkan kemiripan antar item satu sama lainny.

# 
