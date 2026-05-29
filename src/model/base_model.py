from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import json
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    cohen_kappa_score
)

from scipy.stats import mode

import matplotlib.pyplot as plt

# =========================================================
# BASE CLASS
# =========================================================

class MLModel(ABC):

    CONFIG_NAME = "config.json"

    WEIGHTS_NAME = "weights.pkl"

    @classmethod
    @abstractmethod
    def parse_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def export_config(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_weights(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def save_weights(self, path: str | Path) -> None:
        pass

    @classmethod
    def from_folder(cls, folder: str | Path):

        folder = Path(folder)

        with open(folder / cls.CONFIG_NAME, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        parsed = cls.parse_config(raw_config)

        model = cls(**parsed)

        model.load_weights(folder / cls.WEIGHTS_NAME)

        return model

    def save_folder(self, folder: str | Path) -> None:

        folder = Path(folder)

        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / self.CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(self.export_config(), f, indent=4)

        self.save_weights(folder / self.WEIGHTS_NAME)


# =========================================================
# BASE CLUSTERING MODEL
# =========================================================

class ClusteringModel(MLModel, ABC):

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        pass

    @staticmethod
    def preprocess_embeddings(X: np.ndarray) -> np.ndarray:

        X = np.nan_to_num(X)

        X = normalize(X, norm="l2")

        return X

    @staticmethod
    def map_clusters(true_labels: np.ndarray, pseudo_labels: np.ndarray):
        mapped_labels = np.zeros_like(true_labels)

        cluster_label_map = {}

        for cluster_id in np.unique(pseudo_labels):

            mask = pseudo_labels == cluster_id

            majority_label = mode(
                true_labels[mask],
                keepdims=False
            ).mode

            mapped_labels[mask] = majority_label

            cluster_label_map[
                cluster_id
            ] = majority_label

        return (
            mapped_labels,
            cluster_label_map
        )

    @staticmethod
    def evaluate_clustering(
        X: np.ndarray,
        true_labels: np.ndarray,
        pseudo_labels: np.ndarray,
        mapped_labels: np.ndarray
    ) -> dict[str, float]:

        metrics = {}

        # ---------------------------------
        # Internal metrics
        # ---------------------------------

        metrics["silhouette_score"] = silhouette_score(X, pseudo_labels)

        metrics["davies_bouldin_score"] = davies_bouldin_score(X, pseudo_labels)

        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, pseudo_labels)

        except Exception:

            metrics["calinski_harabasz_score"] = np.nan

        # ---------------------------------
        # External metrics
        # ---------------------------------

        metrics["ari_score"] = (adjusted_rand_score(true_labels, pseudo_labels))

        metrics["nmi_score"] = (
            normalized_mutual_info_score(
                true_labels,
                pseudo_labels
            )
        )

        metrics["kappa_score"] = (
            cohen_kappa_score(
                true_labels,
                mapped_labels
            )
        )

        return metrics

    @staticmethod
    def save_cluster_plot(
        X: np.ndarray,
        labels: np.ndarray,
        output_path: str | Path,
        random_state: int = 42
    ):
        pca = PCA(n_components=2, random_state=random_state)

        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            s=10,
            alpha=0.7
        )

        plt.title(
            "Cluster Visualization"
        )

        plt.xlabel(
            "PCA Component 1"
        )

        plt.ylabel(
            "PCA Component 2"
        )

        plt.colorbar(scatter)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()