from __future__ import annotations
from src.model.base_model import ClusteringModel
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Any
import joblib
import numpy as np

class KMeansClustering(ClusteringModel):

    WEIGHTS_NAME = "kmeans.pkl"

    def __init__(self, n_clusters: int, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return  {
                "n_clusters": config["n_clusters"],
                "random_state": config["random_state"],
                "n_init": config["n_init"]
                }

    def export_config(self) -> dict[str, Any]:
        return  {
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
                "n_init": self.n_init
                }

    def save_weights(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    def load_weights(self, path: str | Path) -> None:
        self.model = joblib.load(path)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)