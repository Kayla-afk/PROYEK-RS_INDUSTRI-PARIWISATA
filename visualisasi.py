# visualisasi.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# KONFIGURASI VISUAL
# =========================================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# =========================================================
# LOAD DATASET
# =========================================================
df_places = pd.read_csv(
    "E:/KULIAH/SEMESTER 4/SISTEM REKOMENDASI/PROYEK RS_INDUSTRI PARIWISATA/data/indonesia tourism destination/tourism_with_id.csv"
)

# Handle missing values
for col in ['Category', 'Description', 'City', 'Place_Name']:
    if col not in df_places.columns:
        df_places[col] = ""

    df_places[col] = df_places[col].fillna('')

# =========================================================
# VISUALISASI 1
# DISTRIBUSI KATEGORI WISATA
# =========================================================
plt.figure(figsize=(10, 6))

ax = sns.countplot(
    data=df_places,
    y='Category',
    order=df_places['Category'].value_counts().index,
    hue='Category',
    legend=False,
    palette='viridis'
)

plt.title(
    'Distribusi Kategori Destinasi Wisata di Indonesia',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Jumlah Destinasi', fontsize=12)
plt.ylabel('Kategori', fontsize=12)

plt.tight_layout()

plt.savefig('poster_kategori_wisata.png')

plt.close()

print("Visualisasi 1 (Bar Chart) berhasil disimpan.")

# =========================================================
# VISUALISASI 2
# PETA PERSEBARAN GEOGRAFIS DESTINASI
# =========================================================
plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df_places,
    x='Long',
    y='Lat',
    hue='City',
    palette='Set1',
    s=100,
    alpha=0.8,
    edgecolor='black'
)

plt.title(
    'Pemetaan Titik Lokasi Destinasi Wisata',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

plt.legend(
    title='Kota / Kabupaten',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()

plt.savefig('poster_peta_persebaran.png')

plt.close()

print("Visualisasi 2 (Geospatial Plot) berhasil disimpan.")

# =========================================================
# VISUALISASI 3
# HEATMAP COSINE SIMILARITY
# =========================================================

# Ambil sample destinasi
sample_places = df_places.head(6).copy()

# Gabungkan fitur teks
text_features = (
    sample_places['Place_Name'].astype(str) + ' ' +
    sample_places['Category'].astype(str) + ' ' +
    sample_places['Description'].astype(str) + ' ' +
    sample_places['City'].astype(str)
)

# Text preprocessing sederhana
text_features = text_features.str.lower()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')

# Transform text menjadi matrix
tfidf_matrix = tfidf.fit_transform(text_features)

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Plot Heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    cosine_sim,
    annot=True,
    cmap='YlGnBu',
    xticklabels=sample_places['Place_Name'].values,
    yticklabels=sample_places['Place_Name'].values,
    fmt='.2f',
    linewidths=0.5
)

plt.title(
    'Matriks Cosine Similarity Antar Destinasi',
    fontsize=14,
    fontweight='bold'
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig('poster_cosine_similarity.png')

plt.close()

print("Visualisasi 3 (Heatmap) berhasil disimpan.")

print("Semua visualisasi berhasil dibuat.")
