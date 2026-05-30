from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import *
from src.model.base_model import ClusteringModel
from src.model.kmeans import KMeansClustering
from src.model.mmdcdl import AutoencoderMultimodalClustering
from src.model.umcc import UMCClusteringModel

print("Loading data...")
data = np.load(
    TRAIN_MULTIMODAL_OUTPUT,
    allow_pickle=True
)

X = data["embeddings"].astype(np.float64)

true_labels = data["labels"]

productids = data["productid"]

imageids = data["imageid"]

texts = data["texts"]

print("Preprocessing data...")
X = ClusteringModel.preprocess(X)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_path = "outputs/run_20260529_161646"

run_path = Path(run_path)
if run_path is None:
    run_path = (Path.cwd() / "outputs" / f"run_{timestamp}")
    print(f"Creating folder for outputs in {run_path}...")
    run_path.mkdir(parents=True, exist_ok=True)
else:
    print(f"Using existing outputs folder in {run_path}...")

models_root = run_path / "models"

results_root = run_path / "result"

models_root.mkdir(exist_ok=True)

results_root.mkdir(exist_ok=True)

print("Registering models...")
num_clusters = len(np.unique(true_labels))

MODELS: dict[str, ClusteringModel] = {

    "k_means": KMeansClustering(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10
    ),

    "mmdc": AutoencoderMultimodalClustering(
        input_dim_1=TEXT_DIM,
        input_dim_2=IMAGE_DIM,
        n_clusters=num_clusters
    ),
    
    "umcc": UMCClusteringModel(
        n_clusters=num_clusters
    ),

}

# =========================================================
# TRAIN / EVALUATE LOOP
# =========================================================

for model_name, clustering_model in MODELS.items():

    print(f"\n========== {model_name.upper()} ==========")

    # -----------------------------------------------------
    # MODEL FOLDER
    # -----------------------------------------------------

    model_folder = models_root / model_name

    result_folder = results_root / model_name

    result_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # LOAD / TRAIN
    # -----------------------------------------------------

    if model_folder.exists():

        print(f"Loading existing {model_name} model...")

        clustering_model = clustering_model.from_folder(model_folder)

        pseudo_labels = clustering_model.predict(X)

    else:

        print(f"Training new {model_name} model...")

        pseudo_labels = clustering_model.fit_predict(X)

        clustering_model.save_folder(model_folder)

    print(f"Mapping clusters for model {model_name}...")
    mapped_labels, cluster_map = clustering_model.map_clusters(true_labels, pseudo_labels)

    print(f"Evaluating clusters for model {model_name}")
    metrics = clustering_model.evaluate(X, true_labels, pseudo_labels, mapped_labels)

    print("\n========== METRICS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"Saving {model_name} CSV...")

    df = pd.DataFrame({
        "productid": productids,
        "imageid": imageids,
        "text": texts,
        "true_label": true_labels,
        "pseudo_label": pseudo_labels,
        "mapped_label": mapped_labels
    })

    df.to_csv(result_folder / "cluster_result.csv", index=False)

    print(f"Saving plot for {model_name} model...")
    clustering_model.save_cluster_plot(X, pseudo_labels, result_folder / "cluster_plot.png")

    print(f"Saving {model_name} metrics...")
    metrics_df = pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values())
        })

    metrics_df.to_csv(result_folder / "metrics.csv", index=False)

print("\nDone!")