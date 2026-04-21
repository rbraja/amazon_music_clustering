import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="🎵 Music Clustering App", layout="wide")
st.title("🎵 Amazon Music Clustering (End-to-End ML App)")

# =========================================
# FILE UPLOAD
# =========================================
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    
    
    # 🔥 USER CONTROL: LIMIT DATA SIZE
    #sample_size = st.sidebar.slider("Limit Rows (for low RAM)", 1000, 50000, 5000)

    #df = pd.read_csv(file)

    # Take sample to reduce memory
    #if len(df) > sample_size:
        #df = df.sample(sample_size, random_state=42)

    #st.success(f"Using {len(df)} rows for processing")
    
    df = pd.read_csv(file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # =========================================
    # CLEANING
    # =========================================
    df_clean = df.drop(columns=['track_id', 'track_name', 'artist_name'], errors='ignore')
    df_clean = df_clean.drop_duplicates().dropna()

    st.write("Cleaned Shape:", df_clean.shape)

    # =========================================
    # FEATURES
    # =========================================
    features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms'
    ]

    X = df_clean[features]

    # =========================================
    # SCALING
    # =========================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================
    # SIDEBAR SETTINGS
    # =========================================
    st.sidebar.header("⚙️ Settings")

    use_pca = st.sidebar.checkbox("Use PCA")

    if use_pca:
        n_comp = st.sidebar.slider("PCA Components", 2, 10, 5)
        pca = PCA(n_components=n_comp)
        X_final = pca.fit_transform(X_scaled)
    else:
        X_final = X_scaled

    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Auto Select Best", "K-Means", "DBSCAN", "Hierarchical"]
    )

    # =========================================
    # K-MEANS FUNCTION
    # =========================================
    def run_kmeans(X_data):
        K_range = range(2, 11)
        inertia = []
        silhouette_scores = []

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X_data)

            inertia.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X_data, labels))

        best_k = K_range[np.argmax(silhouette_scores)]

        return best_k, inertia, silhouette_scores

    # =========================================
    # AUTO MODEL SELECTION
    # =========================================
    if model_choice == "Auto Select Best":

        st.subheader("🤖 Auto Model Selection")

        best_k, _, _ = run_kmeans(X_final)

        # KMeans
        km = KMeans(n_clusters=best_k, random_state=42)
        k_labels = km.fit_predict(X_final)
        k_sil = silhouette_score(X_final, k_labels)
        k_db = davies_bouldin_score(X_final, k_labels)

        # DBSCAN
        db = DBSCAN(eps=1.5, min_samples=5)
        d_labels = db.fit_predict(X_final)

        if len(set(d_labels)) > 1:
            d_sil = silhouette_score(X_final, d_labels)
            d_db = davies_bouldin_score(X_final, d_labels)
        else:
            d_sil = -1
            d_db = 999

        # Hierarchical
        agg = AgglomerativeClustering(n_clusters=best_k)
        a_labels = agg.fit_predict(X_final)
        a_sil = silhouette_score(X_final, a_labels)
        a_db = davies_bouldin_score(X_final, a_labels)

        results = pd.DataFrame({
            "Model": ["K-Means", "DBSCAN", "Hierarchical"],
            "Silhouette": [k_sil, d_sil, a_sil],
            "DB Index": [k_db, d_db, a_db]
        })

        st.dataframe(results)

        best_model = results.sort_values(
            by=["Silhouette", "DB Index"],
            ascending=[False, True]
        ).iloc[0]

        st.success(f"Best Model: {best_model['Model']}")

        if best_model["Model"] == "K-Means":
            df_clean['cluster'] = k_labels
        elif best_model["Model"] == "DBSCAN":
            df_clean['cluster'] = d_labels
        else:
            df_clean['cluster'] = a_labels

    # =========================================
    # K-MEANS WITH ELBOW
    # =========================================
    elif model_choice == "K-Means":

        st.subheader("📉 K-Means Optimization")

        best_k, inertia, silhouette_scores = run_kmeans(X_final)

        K_range = range(2, 11)

        # Elbow Plot
        st.write("### 📉 Elbow Method (Inertia vs K)")
        fig1, ax1 = plt.subplots()
        ax1.plot(K_range, inertia, marker='o')
        ax1.set_xlabel("K")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Curve")
        st.pyplot(fig1)

        # Silhouette Plot
        st.write("### 📊 Silhouette Score vs K")
        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, silhouette_scores, marker='o')
        ax2.set_xlabel("K")
        ax2.set_ylabel("Score")
        ax2.set_title("Silhouette Scores")
        st.pyplot(fig2)

        st.success(f"Suggested Best K: {best_k}")

        k = st.slider("Select K", 2, 10, best_k)

        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_final)

        df_clean['cluster'] = labels

        st.write("Silhouette:", silhouette_score(X_final, labels))
        st.write("DB Index:", davies_bouldin_score(X_final, labels))

    # =========================================
    # DBSCAN
    # =========================================
    elif model_choice == "DBSCAN":

        eps = st.sidebar.slider("EPS", 0.5, 5.0, 1.5)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_final)

        df_clean['cluster'] = labels

        if len(set(labels)) > 1:
            st.write("Silhouette:", silhouette_score(X_final, labels))
            st.write("DB Index:", davies_bouldin_score(X_final, labels))
        else:
            st.write("Clustering not meaningful with current parameters")

    # =========================================
    # HIERARCHICAL
    # =========================================
    elif model_choice == "Hierarchical":

        k = st.sidebar.slider("Clusters", 2, 10, 4)

        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_final)

        df_clean['cluster'] = labels

        st.write("Silhouette:", silhouette_score(X_final, labels))
        st.write("DB Index:", davies_bouldin_score(X_final, labels))

    # =========================================
    # VISUALIZATION
    # =========================================
    st.subheader("📊 Cluster Visualization")

    pca2 = PCA(n_components=2)
    vis = pca2.fit_transform(X_scaled)

    df_clean['pca1'] = vis[:, 0]
    df_clean['pca2'] = vis[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df_clean, ax=ax)
    st.pyplot(fig)

    # =========================================
    # INSIGHTS
    # =========================================
    st.subheader("🧠 Cluster Insights")

    summary = df_clean.groupby('cluster')[features].mean()
    st.dataframe(summary)

    # =========================================
    # DOWNLOAD
    # =========================================
    csv = df_clean.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Clustered Data",
        csv,
        "clustered_output.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV file to start")