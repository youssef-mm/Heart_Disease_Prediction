# src/clustering.py
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

DATA_TRAIN = "data/heart_disease_preprocessed.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    df = pd.read_csv(DATA_TRAIN).drop(columns=["num"])

    # K-Means Elbow method
    distortions = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df)
        distortions.append(km.inertia_)

    plt.plot(range(1, 10), distortions, marker='o')
    plt.title("K-Means Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.savefig(f"{RESULTS_DIR}/kmeans_elbow.png")
    plt.close()

    # Hierarchical Clustering
    linked = linkage(df, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.savefig(f"{RESULTS_DIR}/hierarchical_dendrogram.png")
    plt.close()

    print("✅ Clustering analysis completed.")

if __name__ == "__main__":
    run()