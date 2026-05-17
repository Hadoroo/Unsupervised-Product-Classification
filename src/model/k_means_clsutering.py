import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "train_multimodal_embeddings.npz"

OUTPUT_CSV = "clustering_result.csv"

OUTPUT_NPZ = "clustered_multimodal_embeddings.npz"

RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

X = data["embeddings"]

true_labels = data["labels"]

productids = data["productid"]

imageids = data["imageid"]

texts = data["texts"]


# =========================================================
# INFO
# =========================================================

print("Embedding shape:")
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
# KMEANS CLUSTERING
# =========================================================

print("\nTraining KMeans...")

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=RANDOM_STATE,
    n_init=10
)

pseudo_labels = kmeans.fit_predict(X)

print("Clustering finished")


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

ch_score = calinski_harabasz_score(
    X,
    pseudo_labels
)


# -----------------------------------------
# External Metrics
# (because user has true labels)
# -----------------------------------------

ari_score = adjusted_rand_score(
    true_labels,
    pseudo_labels
)

nmi_score = normalized_mutual_info_score(
    true_labels,
    pseudo_labels
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


# =========================================================
# CLUSTER DISTRIBUTION
# =========================================================

cluster_counts = pd.Series(
    pseudo_labels
).value_counts().sort_index()

print("\n========== CLUSTER DISTRIBUTION ==========")

print(cluster_counts)


# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame({

    "productid": productids,

    "imageid": imageids,

    "text": texts,

    "true_label": true_labels,

    "pseudo_label": pseudo_labels
})

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"\nSaved CSV: {OUTPUT_CSV}")


# =========================================================
# SAVE NPZ
# =========================================================

np.savez(

    OUTPUT_NPZ,

    embeddings=X,

    true_labels=true_labels,

    pseudo_labels=pseudo_labels,

    productid=productids,

    imageid=imageids,

    texts=texts
)

print(f"Saved NPZ: {OUTPUT_NPZ}")


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
    "pseudo_label",
    "text"
]])


print("\nDone!")