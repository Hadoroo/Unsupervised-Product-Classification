import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "clustered_multimodal_embeddings.npz"

RANDOM_STATE = 42

TSNE_SAMPLE_SIZE = 5000


# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

X = data["embeddings"]

true_labels = data["true_labels"]

pseudo_labels = data["pseudo_labels"]

texts = data["texts"]

print("Embedding shape:")
print(X.shape)

print("\nTotal samples:")
print(len(X))


# =========================================================
# INTERNAL METRICS
# =========================================================

print("\n===================================")
print("INTERNAL CLUSTER METRICS")
print("===================================")

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

print(f"Silhouette Score         : {sil_score:.4f}")

print(f"Davies-Bouldin Score     : {db_score:.4f}")

print(f"Calinski-Harabasz Score  : {ch_score:.4f}")


# =========================================================
# EXTERNAL METRICS
# =========================================================

print("\n===================================")
print("EXTERNAL CLUSTER METRICS")
print("===================================")

ari_score = adjusted_rand_score(
    true_labels,
    pseudo_labels
)

nmi_score = normalized_mutual_info_score(
    true_labels,
    pseudo_labels
)

print(f"Adjusted Rand Index      : {ari_score:.4f}")

print(f"Normalized Mutual Info   : {nmi_score:.4f}")


# =========================================================
# CLUSTER DISTRIBUTION
# =========================================================

print("\n===================================")
print("CLUSTER DISTRIBUTION")
print("===================================")

cluster_counts = pd.Series(
    pseudo_labels
).value_counts().sort_index()

print(cluster_counts)


# =========================================================
# SAVE DISTRIBUTION
# =========================================================

cluster_counts.to_csv(
    "cluster_distribution.csv"
)

print("\nCluster distribution saved")


# =========================================================
# SILHOUETTE PER SAMPLE
# =========================================================

print("\nCalculating silhouette samples...")

sample_scores = silhouette_samples(
    X,
    pseudo_labels
)

silhouette_df = pd.DataFrame({

    "pseudo_label": pseudo_labels,

    "silhouette_score": sample_scores
})

silhouette_df.to_csv(
    "silhouette_scores.csv",
    index=False
)

print("Silhouette scores saved")


# =========================================================
# PCA VISUALIZATION
# =========================================================

print("\nRunning PCA...")

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pseudo_labels,
    s=5
)

plt.title("PCA Cluster Visualization")

plt.xlabel("PCA-1")

plt.ylabel("PCA-2")

plt.savefig(
    "pca_cluster_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("PCA visualization saved")


# =========================================================
# TSNE VISUALIZATION
# =========================================================

print("\nPreparing t-SNE...")

# Sampling supaya t-SNE tidak terlalu berat
if len(X) > TSNE_SAMPLE_SIZE:

    idx = np.random.choice(
        len(X),
        TSNE_SAMPLE_SIZE,
        replace=False
    )

    X_tsne_input = X[idx]

    pseudo_tsne = pseudo_labels[idx]

else:

    X_tsne_input = X

    pseudo_tsne = pseudo_labels


print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_STATE,
    init="pca"
)

X_tsne = tsne.fit_transform(
    X_tsne_input
)

plt.figure(figsize=(10, 8))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=pseudo_tsne,
    s=5
)

plt.title("t-SNE Cluster Visualization")

plt.xlabel("tSNE-1")

plt.ylabel("tSNE-2")

plt.savefig(
    "tsne_cluster_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("t-SNE visualization saved")


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    true_labels,
    pseudo_labels
)

cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    "cluster_confusion_matrix.csv",
    index=False
)

print("Confusion matrix saved")


# =========================================================
# SAMPLE CLUSTER INSPECTION
# =========================================================

print("\n===================================")
print("SAMPLE CLUSTER INSPECTION")
print("===================================")

inspection_df = pd.DataFrame({

    "text": texts,

    "true_label": true_labels,

    "pseudo_label": pseudo_labels
})

sample_cluster = 0

sample_data = inspection_df[
    inspection_df["pseudo_label"] == sample_cluster
].head(10)

print(sample_data)


# =========================================================
# SUMMARY REPORT
# =========================================================

summary = pd.DataFrame({

    "metric": [

        "Silhouette Score",

        "Davies-Bouldin Score",

        "Calinski-Harabasz Score",

        "Adjusted Rand Index",

        "Normalized Mutual Information"
    ],

    "value": [

        sil_score,

        db_score,

        ch_score,

        ari_score,

        nmi_score
    ]
})

summary.to_csv(
    "cluster_evaluation_summary.csv",
    index=False
)

print("\nEvaluation summary saved")


# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("EVALUATION FINISHED")
print("===================================")