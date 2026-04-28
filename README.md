# 🎵 Amazon Music Clustering (Streamlit ML App)

## 📌 Overview
This project builds an **unsupervised machine learning system** to automatically group songs based on their audio features such as energy, danceability, tempo, and more.

It uses clustering techniques to discover patterns in music data without labeled genres and provides an interactive UI using Streamlit.

---

## 🚀 Features

- 📂 Upload your own dataset (CSV)
- ⚙️ Data preprocessing & scaling
- 🔍 PCA (optional dimensionality reduction)
- 🤖 Multiple clustering algorithms:
  - K-Means (with Elbow Method & Silhouette Score)
  - DBSCAN
  - Hierarchical Clustering
- 🧠 Auto model selection based on evaluation metrics
- 📊 Cluster visualization (2D PCA)
- 📈 Cluster insights (feature averages)
- ⬇️ Download clustered dataset

---

## 🧠 Problem Statement

Manual categorization of songs into genres is not scalable.  
This project uses clustering to group similar songs automatically based on audio characteristics.

---

## 💼 Business Use Cases

- 🎧 Personalized playlist generation  
- 🔎 Song recommendation systems  
- 🎤 Artist and competitor analysis  
- 📊 User segmentation for music platforms  

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  

---

## 📊 Clustering Techniques Used

### 🔹 K-Means
- Uses centroid-based clustering  
- Optimal K selected using:
  - Elbow Method  
  - Silhouette Score  

### 🔹 DBSCAN
- Density-based clustering  
- Detects noise and outliers  

### 🔹 Hierarchical Clustering
- Builds cluster hierarchy  
- Uses linkage methods  

---

## 📏 Evaluation Metrics

- Silhouette Score  
- Davies-Bouldin Index  
- Cluster Visualization  

---

## 📁 Dataset Features

- danceability  
- energy  
- loudness  
- speechiness  
- acousticness  
- instrumentalness  
- liveness  
- valence  
- tempo  
- duration_ms  

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
