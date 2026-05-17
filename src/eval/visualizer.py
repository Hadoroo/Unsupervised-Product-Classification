import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import umap.umap_ as umap


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "clustered_multimodal_embeddings.npz"

RANDOM_STATE = 42

MAX_SAMPLE = 5000


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
# SAMPLING
# =========================================================

# Visualisasi terlalu berat jika data besar
# maka gunakan sampling

if len(X) > MAX_SAMPLE:

    idx = np.random.choice(
        len(X),
        MAX_SAMPLE,
        replace=False
    )

    X_vis = X[idx]

    true_vis = true_labels[idx]

    pseudo_vis = pseudo_labels[idx]

    texts_vis = texts[idx]

else:

    X_vis = X

    true_vis = true_labels

    pseudo_vis = pseudo_labels

    texts_vis = texts


print(f"\nVisualization samples: {len(X_vis)}")


# =========================================================
# PCA VISUALIZATION
# =========================================================

print("\nRunning PCA...")

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(X_vis)

print("PCA done")


# =========================================================
# PLOT PCA PSEUDO LABEL
# =========================================================

plt.figure(figsize=(12, 10))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pseudo_vis,
    s=5
)

plt.title(
    "PCA Visualization - Pseudo Labels"
)

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.savefig(
    "pca_pseudo_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: pca_pseudo_labels.png")


# =========================================================
# PLOT PCA TRUE LABEL
# =========================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=true_vis,
    s=5
)

plt.title(
    "PCA Visualization - True Labels"
)

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.savefig(
    "pca_true_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: pca_true_labels.png")


# =========================================================
# TSNE
# =========================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(X_vis)

print("t-SNE done")


# =========================================================
# TSNE PSEUDO LABEL
# =========================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=pseudo_vis,
    s=5
)

plt.title(
    "t-SNE Visualization - Pseudo Labels"
)

plt.xlabel("tSNE-1")

plt.ylabel("tSNE-2")

plt.savefig(
    "tsne_pseudo_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: tsne_pseudo_labels.png")


# =========================================================
# TSNE TRUE LABEL
# =========================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=true_vis,
    s=5
)

plt.title(
    "t-SNE Visualization - True Labels"
)

plt.xlabel("tSNE-1")

plt.ylabel("tSNE-2")

plt.savefig(
    "tsne_true_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: tsne_true_labels.png")


# =========================================================
# UMAP
# =========================================================

print("\nRunning UMAP...")

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=RANDOM_STATE
)

X_umap = umap_model.fit_transform(X_vis)

print("UMAP done")


# =========================================================
# UMAP PSEUDO LABEL
# =========================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=pseudo_vis,
    s=5
)

plt.title(
    "UMAP Visualization - Pseudo Labels"
)

plt.xlabel("UMAP-1")

plt.ylabel("UMAP-2")

plt.savefig(
    "umap_pseudo_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: umap_pseudo_labels.png")


# =========================================================
# UMAP TRUE LABEL
# =========================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=true_vis,
    s=5
)

plt.title(
    "UMAP Visualization - True Labels"
)

plt.xlabel("UMAP-1")

plt.ylabel("UMAP-2")

plt.savefig(
    "umap_true_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: umap_true_labels.png")


# =========================================================
# SAVE EMBEDDING VISUALIZATION
# =========================================================

vis_df = pd.DataFrame({

    "pca_x": X_pca[:, 0],
    "pca_y": X_pca[:, 1],

    "tsne_x": X_tsne[:, 0],
    "tsne_y": X_tsne[:, 1],

    "umap_x": X_umap[:, 0],
    "umap_y": X_umap[:, 1],

    "true_label": true_vis,

    "pseudo_label": pseudo_vis,

    "text": texts_vis
})

vis_df.to_csv(
    "embedding_visualization_coordinates.csv",
    index=False
)

print("\nSaved: embedding_visualization_coordinates.csv")


# =========================================================
# CLUSTER SAMPLE INSPECTION
# =========================================================

print("\n===================================")
print("SAMPLE CLUSTER INSPECTION")
print("===================================")

sample_cluster = 0

cluster_df = vis_df[
    vis_df["pseudo_label"] == sample_cluster
].head(10)

print(cluster_df[[
    "true_label",
    "pseudo_label",
    "text"
]])


# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("VISUALIZATION FINISHED")
print("===================================")