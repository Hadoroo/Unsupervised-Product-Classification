"""
Autoencoder-based Deep Multimodal Clustering Model
====================================================
Berdasarkan paper:
    "Multi-modal data clustering using deep learning: A systematic review"
    Raya et al., Neurocomputing 607 (2024) 128348

Pendekatan yang diimplementasikan mengacu pada framework DMMC (Zhang et al., 2020)
dan DCUMC (Zong et al., 2020) yang dibahas di Section 6.1.1 paper tersebut.

Arsitektur:
    - Dua Autoencoder terpisah per modalitas (modality-specific learning)
    - Fusion layer untuk menggabungkan representasi laten
    - K-Means clustering pada representasi yang telah di-fuse
    - Cross-reconstruction loss untuk menyelaraskan fitur antar modalitas

Referensi dari paper (Section 6.1.1):
    - DMMC: "employs two AEs, each handling one modality, and integrates
      clustering through a fusion layer."
    - DCUMC: "extracting shared and unique features per modality using an AE
      structure, facilitating cross-reconstruction of modality features."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.model.base_model import ClusteringModel


# =========================================================
# ENCODER & DECODER SUB-NETWORKS
# =========================================================

class ModalityEncoder(nn.Module):
    """
    Encoder untuk satu modalitas.
    Memetakan input berdimensi tinggi ke ruang laten berdimensi rendah.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModalityDecoder(nn.Module):
    """
    Decoder untuk satu modalitas.
    Merekonstruksi input asli dari representasi laten.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()

        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class FusionLayer(nn.Module):
    """
    Fusion layer untuk menggabungkan representasi laten dari dua modalitas.
    Menghasilkan representasi bersama (shared representation) untuk clustering.
    Sesuai dengan pendekatan 'early fusion' pada representasi laten
    yang dibahas di Section 5.4 paper.
    """

    def __init__(self, latent_dim_1: int, latent_dim_2: int, fused_dim: int):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim_1 + latent_dim_2, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([z1, z2], dim=-1)
        return self.fusion(combined)


# =========================================================
# AUTOENCODER MULTIMODAL CLUSTERING MODEL
# =========================================================

class AutoencoderMultimodalClustering(ClusteringModel):
    """
    Deep Multimodal Clustering berbasis Autoencoder dengan dua modalitas.

    Mengimplementasikan pendekatan yang dibahas dalam survey (Section 6.1.1):
    - Dua AE terpisah untuk masing-masing modalitas (modality-specific learning)
    - Cross-reconstruction loss untuk memaksa representasi laten saling
      mengandung informasi lintas modalitas
    - Fusion layer menghasilkan representasi bersama untuk K-Means clustering

    Loss total:
        L = L_recon_1 + L_recon_2 + alpha * L_cross_recon + beta * L_cluster

    di mana:
        L_recon        : reconstruction loss per modalitas
        L_cross_recon  : cross-modal reconstruction loss
        L_cluster      : clustering loss (jarak ke centroid K-Means)
    """

    WEIGHTS_NAME = "weights"
    
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        latent_dim: int = 64,
        fused_dim: int = 128,
        hidden_dims: list[int] | None = None,
        n_clusters: int = 10,
        alpha: float = 0.5,
        beta: float = 0.1,
        lr: float = 1e-3,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.latent_dim = latent_dim
        self.fused_dim = fused_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.n_clusters = n_clusters
        self.alpha = alpha          # bobot cross-reconstruction loss
        self.beta = beta            # bobot clustering loss
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        torch.manual_seed(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_networks()

        # K-Means untuk fase clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

        # Cluster centroids (diinisialisasi setelah pretrain)
        self.cluster_centers_: np.ndarray | None = None

    # ---------------------------------------------------------
    # Network construction
    # ---------------------------------------------------------

    def _build_networks(self) -> None:
        """Membangun semua sub-jaringan AE dan fusion layer."""

        # Encoder modalitas 1
        self.encoder_1 = ModalityEncoder(
            self.input_dim_1, self.latent_dim, self.hidden_dims
        ).to(self.device)

        # Decoder modalitas 1 (rekonstruksi normal)
        self.decoder_1 = ModalityDecoder(
            self.latent_dim, self.input_dim_1, self.hidden_dims
        ).to(self.device)

        # Encoder modalitas 2
        self.encoder_2 = ModalityEncoder(
            self.input_dim_2, self.latent_dim, self.hidden_dims
        ).to(self.device)

        # Decoder modalitas 2 (rekonstruksi normal)
        self.decoder_2 = ModalityDecoder(
            self.latent_dim, self.input_dim_2, self.hidden_dims
        ).to(self.device)

        # Cross-decoder: z1 -> rekonstruksi modalitas 2 (cross-reconstruction)
        self.cross_decoder_1to2 = ModalityDecoder(
            self.latent_dim, self.input_dim_2, self.hidden_dims
        ).to(self.device)

        # Cross-decoder: z2 -> rekonstruksi modalitas 1 (cross-reconstruction)
        self.cross_decoder_2to1 = ModalityDecoder(
            self.latent_dim, self.input_dim_1, self.hidden_dims
        ).to(self.device)

        # Fusion layer
        self.fusion = FusionLayer(
            self.latent_dim, self.latent_dim, self.fused_dim
        ).to(self.device)

    def _get_all_params(self) -> list:
        return (
            list(self.encoder_1.parameters())
            + list(self.decoder_1.parameters())
            + list(self.encoder_2.parameters())
            + list(self.decoder_2.parameters())
            + list(self.cross_decoder_1to2.parameters())
            + list(self.cross_decoder_2to1.parameters())
            + list(self.fusion.parameters())
        )

    # ---------------------------------------------------------
    # Loss functions
    # ---------------------------------------------------------

    def _reconstruction_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(x_hat, x)

    def _clustering_loss(
        self, z_fused: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft clustering loss: mendorong representasi mendekati centroid terdekat.
        Mirip dengan soft assignment pada DEC (Xie et al., 2016).
        """
        # Hitung jarak ke semua centroid
        dists = torch.cdist(z_fused, centers)           # (B, K)
        min_dists, _ = dists.min(dim=1)                 # (B,)
        return min_dists.mean()

    # ---------------------------------------------------------
    # Pre-training (AE saja, tanpa clustering loss)
    # ---------------------------------------------------------

    def _pretrain(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
    ) -> None:
        """
        Fase pretrain: latih AE dengan reconstruction + cross-reconstruction loss.
        Sesuai dengan langkah pertama DCUMC (Section 6.1.1 paper).
        """
        optimizer = optim.Adam(self._get_all_params(), lr=self.lr)
        dataset = TensorDataset(X1, X2)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._set_train_mode(True)

        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            for x1_batch, x2_batch in loader:
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)

                z1 = self.encoder_1(x1_batch)
                z2 = self.encoder_2(x2_batch)

                # Reconstruction loss
                loss_r1 = self._reconstruction_loss(x1_batch, self.decoder_1(z1))
                loss_r2 = self._reconstruction_loss(x2_batch, self.decoder_2(z2))

                # Cross-reconstruction loss
                loss_c12 = self._reconstruction_loss(
                    x2_batch, self.cross_decoder_1to2(z1)
                )
                loss_c21 = self._reconstruction_loss(
                    x1_batch, self.cross_decoder_2to1(z2)
                )

                loss = (
                    loss_r1 + loss_r2
                    + self.alpha * (loss_c12 + loss_c21)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(loader)
                print(f"  [Pretrain] Epoch {epoch+1}/{self.pretrain_epochs} | Loss: {avg:.4f}")

    # ---------------------------------------------------------
    # Fine-tuning (AE + clustering loss)
    # ---------------------------------------------------------

    def _finetune(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
    ) -> None:
        """
        Fase fine-tune: gabungkan reconstruction loss dengan clustering loss.
        Sesuai dengan langkah kedua DCUMC/DMMC (Section 6.1.1 paper).
        """
        optimizer = optim.Adam(self._get_all_params(), lr=self.lr * 0.5)
        dataset = TensorDataset(X1, X2)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        centers = torch.tensor(
            self.cluster_centers_, dtype=torch.float32
        ).to(self.device)

        self._set_train_mode(True)

        for epoch in range(self.finetune_epochs):
            total_loss = 0.0
            for x1_batch, x2_batch in loader:
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)

                z1 = self.encoder_1(x1_batch)
                z2 = self.encoder_2(x2_batch)
                z_fused = self.fusion(z1, z2)

                loss_r1 = self._reconstruction_loss(x1_batch, self.decoder_1(z1))
                loss_r2 = self._reconstruction_loss(x2_batch, self.decoder_2(z2))
                loss_c12 = self._reconstruction_loss(
                    x2_batch, self.cross_decoder_1to2(z1)
                )
                loss_c21 = self._reconstruction_loss(
                    x1_batch, self.cross_decoder_2to1(z2)
                )
                loss_cluster = self._clustering_loss(z_fused, centers)

                loss = (
                    loss_r1 + loss_r2
                    + self.alpha * (loss_c12 + loss_c21)
                    + self.beta * loss_cluster
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(loader)
                print(f"  [Finetune] Epoch {epoch+1}/{self.finetune_epochs} | Loss: {avg:.4f}")

    # ---------------------------------------------------------
    # Embedding extraction
    # ---------------------------------------------------------

    @torch.no_grad()
    def _get_fused_embeddings(
        self, X1: torch.Tensor, X2: torch.Tensor
    ) -> np.ndarray:
        self._set_train_mode(False)
        dataset = TensorDataset(X1, X2)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_z = []
        for x1_batch, x2_batch in loader:
            z1 = self.encoder_1(x1_batch.to(self.device))
            z2 = self.encoder_2(x2_batch.to(self.device))
            z_fused = self.fusion(z1, z2)
            all_z.append(z_fused.cpu().numpy())
        return np.vstack(all_z)

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def _set_train_mode(self, mode: bool) -> None:
        for net in [
            self.encoder_1, self.decoder_1,
            self.encoder_2, self.decoder_2,
            self.cross_decoder_1to2, self.cross_decoder_2to1,
            self.fusion,
        ]:
            net.train(mode)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = np.nan_to_num(X).astype(np.float32)
        X = normalize(X, norm="l2")
        return torch.tensor(X, dtype=torch.float32)

    def _split_modalities(
        self, X: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Memisahkan input gabungan X menjadi dua modalitas.
        X diasumsikan adalah [modalitas_1 | modalitas_2] yang di-concatenate.
        """
        X1 = X[:, : self.input_dim_1]
        X2 = X[:, self.input_dim_1 :]
        return self._to_tensor(X1), self._to_tensor(X2)

    # ---------------------------------------------------------
    # ClusteringModel interface
    # ---------------------------------------------------------

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Training penuh: pretrain AE → inisialisasi K-Means → finetune → prediksi.

        Args:
            X: Array (N, input_dim_1 + input_dim_2) — dua modalitas di-concatenate.

        Returns:
            Pseudo-labels hasil clustering, shape (N,).
        """
        print("[AutoencoderMultimodalClustering] Memulai training...")

        X1_t, X2_t = self._split_modalities(X)

        # --- Fase 1: Pretrain AE ---
        print("  Fase 1: Pre-training Autoencoder...")
        self._pretrain(X1_t, X2_t)

        # --- Inisialisasi K-Means pada representasi laten ---
        print("  Inisialisasi cluster dengan K-Means...")
        Z_init = self._get_fused_embeddings(X1_t, X2_t)
        self.kmeans.fit(Z_init)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # --- Fase 2: Fine-tune dengan clustering loss ---
        print("  Fase 2: Fine-tuning dengan clustering loss...")
        self._finetune(X1_t, X2_t)

        # --- Prediksi akhir ---
        Z_final = self._get_fused_embeddings(X1_t, X2_t)
        labels = self.kmeans.predict(Z_final)

        print("[AutoencoderMultimodalClustering] Training selesai.")
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi pada data baru menggunakan model yang sudah dilatih.

        Args:
            X: Array (N, input_dim_1 + input_dim_2).

        Returns:
            Pseudo-labels, shape (N,).
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model belum dilatih. Panggil fit_predict() terlebih dahulu.")

        X1_t, X2_t = self._split_modalities(X)
        Z = self._get_fused_embeddings(X1_t, X2_t)
        return self.kmeans.predict(Z)

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------

    def save_weights(self, folder: str | Path) -> None:

        folder = Path(folder)

        folder.mkdir(parents=True, exist_ok=True)

        # ---------------------------------
        # Torch state
        # ---------------------------------

        torch_state = {
            "encoder_1": self.encoder_1.state_dict(),
            "decoder_1": self.decoder_1.state_dict(),
            "encoder_2": self.encoder_2.state_dict(),
            "decoder_2": self.decoder_2.state_dict(),
            "cross_decoder_1to2": self.cross_decoder_1to2.state_dict(),
            "cross_decoder_2to1": self.cross_decoder_2to1.state_dict(),
            "fusion": self.fusion.state_dict(),
            "cluster_centers_": torch.tensor(self.cluster_centers_, dtype=torch.float32)
        }

        torch.save(torch_state, folder / "model.pt")

        # ---------------------------------
        # sklearn object
        # ---------------------------------

        joblib.dump(self.kmeans, folder / "kmeans.pkl")

    def load_weights(self, folder: str | Path) -> None:

        folder = Path(folder)

        state = torch.load(folder / "model.pt", map_location=self.device)

        self.encoder_1.load_state_dict(state["encoder_1"])

        self.decoder_1.load_state_dict(state["decoder_1"])

        self.encoder_2.load_state_dict(state["encoder_2"])

        self.decoder_2.load_state_dict(state["decoder_2"])

        self.cross_decoder_1to2.load_state_dict(state["cross_decoder_1to2"])

        self.cross_decoder_2to1.load_state_dict(state["cross_decoder_2to1"])

        self.fusion.load_state_dict(state["fusion"])

        self.cluster_centers_ = state["cluster_centers_"].cpu().numpy()

        self.kmeans = joblib.load(folder / "kmeans.pkl")

    # ---------------------------------------------------------
    # Config (parse / export)
    # ---------------------------------------------------------

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_dim_1":      config["input_dim_1"],
            "input_dim_2":      config["input_dim_2"],
            "latent_dim":       config["latent_dim"],
            "fused_dim":        config["fused_dim"],
            "hidden_dims":      config["hidden_dims"],
            "n_clusters":       config["n_clusters"],
            "alpha":            config["alpha"],
            "beta":             config["beta"],
            "lr":               config["lr"],
            "pretrain_epochs":  config["pretrain_epochs"],
            "finetune_epochs":  config["finetune_epochs"],
            "batch_size":       config["batch_size"],
            "random_state":     config["random_state"],
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "input_dim_1":      self.input_dim_1,
            "input_dim_2":      self.input_dim_2,
            "latent_dim":       self.latent_dim,
            "fused_dim":        self.fused_dim,
            "hidden_dims":      self.hidden_dims,
            "n_clusters":       self.n_clusters,
            "alpha":            self.alpha,
            "beta":             self.beta,
            "lr":               self.lr,
            "pretrain_epochs":  self.pretrain_epochs,
            "finetune_epochs":  self.finetune_epochs,
            "batch_size":       self.batch_size,
            "random_state":     self.random_state,
        }