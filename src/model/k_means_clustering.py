import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    cohen_kappa_score
)

from sklearn.preprocessing import normalize

from src.config import *

from scipy.stats import mode

# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    TRAIN_MULTIMODAL_OUTPUT,
    allow_pickle=True
)

X = data["embeddings"].astype(np.float64)

true_labels = data["labels"]

productids = data["productid"]

imageids = data["imageid"]

texts = data["texts"]


# =========================================================
# PREPROCESS
# =========================================================

print("\nPreprocessing embeddings...")

# remove nan/inf
X = np.nan_to_num(X)

# normalize
X = normalize(X, norm="l2")

print("Preprocessing finished")


# =========================================================
# INFO
# =========================================================

print("\nEmbedding shape:")
print(X.shape)

print("\nTotal samples:")
print(len(X))


# =========================================================
# TOTAL CLUSTERS
# =========================================================

num_clusters = len(
    np.unique(true_labels)
)

print(f"\nNumber of clusters: {num_clusters}")


# =========================================================
# LOAD OR TRAIN KMEANS
# =========================================================

if os.path.exists(KMEANS_MODEL_PATH):

    print("\nKMeans model already exists")

    print(f"Loading model: {KMEANS_MODEL_PATH}")

    kmeans:KMeans = joblib.load(KMEANS_MODEL_PATH)

    pseudo_labels = kmeans.predict(X)

else:

    print("\nTraining KMeans...")

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    pseudo_labels = kmeans.fit_predict(X)

    print("Clustering finished")

    joblib.dump(
        kmeans,
        KMEANS_MODEL_PATH
    )

    print(f"Saved KMeans model: {KMEANS_MODEL_PATH}")

# =========================================================
# MAP CLUSTERS TO TRUE LABELS
# =========================================================

print("\nMapping clusters to true labels...")

mapped_labels = np.zeros_like(true_labels)

cluster_label_map = {}

for cluster_id in np.unique(pseudo_labels):

    # semua sample pada cluster ini
    mask = pseudo_labels == cluster_id

    # cari true label paling dominan
    majority_label = mode(
        true_labels[mask],
        keepdims=False
    ).mode

    # assign
    mapped_labels[mask] = majority_label

    # simpan mapping
    cluster_label_map[cluster_id] = majority_label

print("Cluster mapping completed")

print("\nCluster Mapping:")

for cluster_id, label_id in cluster_label_map.items():

    print(
        f"Cluster {cluster_id}"
        f" -> True Label {label_id}"
    )

# =========================================================
# EVALUATION
# =========================================================

print("\nCalculating metrics...")


# -----------------------------------------
# Internal Clustering Metrics
# -----------------------------------------

sil_score = silhouette_score(
    X,
    pseudo_labels
)

db_score = davies_bouldin_score(
    X,
    pseudo_labels
)

try:

    ch_score = calinski_harabasz_score(
        X,
        pseudo_labels
    )

except Exception as e:

    print(f"CH Score failed: {e}")

    ch_score = np.nan


# -----------------------------------------
# External Metrics
# -----------------------------------------

ari_score = adjusted_rand_score(
    true_labels,
    pseudo_labels
)

nmi_score = normalized_mutual_info_score(
    true_labels,
    pseudo_labels
)

kappa_score = cohen_kappa_score(
    true_labels,
    mapped_labels
)

# =========================================================
# PRINT RESULT
# =========================================================

print("\n========== CLUSTERING RESULT ==========")

print(f"Silhouette Score           : {sil_score:.4f}")

print(f"Davies-Bouldin Score       : {db_score:.4f}")

print(f"Calinski-Harabasz Score    : {ch_score:.4f}")

print(f"Adjusted Rand Index (ARI)  : {ari_score:.4f}")

print(f"Normalized Mutual Info     : {nmi_score:.4f}")

print(f"Cohen Kappa Score          : {kappa_score:.4f}")

# =========================================================
# CLUSTER DISTRIBUTION
# =========================================================

cluster_counts = pd.Series(
    pseudo_labels
).value_counts().sort_index()

print("\n========== CLUSTER DISTRIBUTION ==========")

print(cluster_counts)


# =========================================================
# SAVE METRICS MARKDOWN
# =========================================================

metrics_md = f"""
# Clustering Evaluation Report

## Clustering Metrics

| Metric | Score |
|---|---:|
| Silhouette Score | {sil_score:.6f} |
| Davies-Bouldin Score | {db_score:.6f} |
| Calinski-Harabasz Score | {ch_score:.6f} |
| Adjusted Rand Index (ARI) | {ari_score:.6f} |
| Normalized Mutual Information (NMI) | {nmi_score:.6f} |
| Cohen Kappa Score | {kappa_score:.6f} |

---

## Dataset Information

| Property | Value |
|---|---:|
| Total Samples | {len(X)} |
| Embedding Dimension | {X.shape[1]} |
| Number of Clusters | {num_clusters} |

---

## Cluster Distribution

{cluster_counts.to_markdown()}

---

## Notes

- Embeddings were normalized using L2 normalization
- Clustering algorithm: KMeans
- Number of initialization runs: 10
- Random state: {RANDOM_STATE}
"""

with open(
    METRIC_REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(metrics_md)

print(f"\nSaved metric report: {METRIC_REPORT_PATH}")

# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame({

    "productid": productids,

    "imageid": imageids,

    "text": texts,

    "true_label": true_labels,

    "pseudo_label": pseudo_labels,

    "mapped_label": mapped_labels
})

df.to_csv(
    CSV_OUTPUT,
    index=False
)

print(f"\nSaved CSV: {CSV_OUTPUT}")


# =========================================================
# SAVE NPZ
# =========================================================

if os.path.exists(MULTIMODAL_CLUSTER_OUTPUT):
    print(f"\nNPZ already exists, skipping save: {MULTIMODAL_CLUSTER_OUTPUT}")

else:

    np.savez(

        MULTIMODAL_CLUSTER_OUTPUT,

        embeddings=X,

        true_labels=true_labels,

        pseudo_labels=pseudo_labels,

        productid=productids,

        imageid=imageids,

        texts=texts
    )

    print(f"\nSaved NPZ: {MULTIMODAL_CLUSTER_OUTPUT}")


# =========================================================
# PCA CLUSTER VISUALIZATION
# =========================================================

print("\nGenerating cluster plot...")

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_2d = pca.fit_transform(X)

plt.figure(figsize=(10, 8))

scatter = plt.scatter(

    X_2d[:, 0],
    X_2d[:, 1],

    c=pseudo_labels,

    s=10,

    alpha=0.7
)

plt.title("KMeans Cluster Visualization (PCA 2D)")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(
    CLUSTER_PLOT_PATH,
    dpi=300
)

plt.close()

print(f"Saved cluster plot: {CLUSTER_PLOT_PATH}")


# =========================================================
# SAMPLE CLUSTER INSPECTION
# =========================================================

print("\n========== SAMPLE CLUSTER ==========")

sample_cluster = 0

sample_df = df[
    df["pseudo_label"] == sample_cluster
].head(10)

print(sample_df[[
    "productid",
    "true_label",
    "mapped_label",
    "pseudo_label",
    "text"
]])


print("\nDone!")