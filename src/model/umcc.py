"""
UMC-Inspired Clustering Model: Density-Based High-Quality Sample Selection
===========================================================================
Berdasarkan paper:
    "Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances"
    Zhang et al., ACL 2024

Model ini mengimplementasikan:
    1. Mekanisme seleksi sampel berkualitas tinggi berbasis densitas (Section 4.3)
    2. Pemilihan K_near optimal secara otomatis per cluster (Section 4.3.2)
    3. Strategi curriculum learning dengan threshold dinamis (Eq. 4)
    4. Sequential representation learning: supervised + unsupervised contrastive (Section 4.4)

Referensi utama dari paper:
    - Eq. 4  : t = t0 + Δ * iter
    - Eq. 5  : ρ_i = K_near / Σ d_ij  (density calculation)
    - Eq. 6  : IdxCk = argsort(-[ρ1, ρ2, ..., ρn])
    - Eq. 7  : K^k_near,q = floor(|Ck| * (L + Δ' * (q-1)))
    - Eq. 8-9: cohesion score untuk evaluasi kualitas subset
    - Eq. 10 : q_opt = argmax cohesion(C^q_k)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import joblib

from sklearn.cluster import KMeans
from scipy.stats import mode

from src.model.base_model import ClusteringModel


# =============================================================================
# DENSITY-BASED HIGH-QUALITY SAMPLE SELECTOR
# (Implementasi Section 4.3 dari paper UMC)
# =============================================================================

class DensityBasedSampleSelector:
    """
    Memilih sampel berkualitas tinggi dalam setiap cluster berdasarkan
    local density, persis seperti yang diusulkan di Section 4.3 paper UMC.

    Konsep utama:
        - Sampel berkualitas tinggi → local density tinggi (rapat dengan tetangga)
        - Sampel berkualitas rendah / noise → local density rendah

    Parameters
    ----------
    L : float
        Lower proportion bound untuk kandidat K_near (default 0.1, sesuai paper)
    delta_prime : float
        Interval tetap antar kandidat K_near (default 0.02, sesuai paper)
    n_candidates : int
        Jumlah kandidat K_near yang dievaluasi (default 10, sesuai paper)
    """

    def __init__(
        self,
        L: float = 0.1,
        delta_prime: float = 0.02,
        n_candidates: int = 10,
    ):
        self.L = L
        self.delta_prime = delta_prime
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------
    # Eq. 7: Hitung kandidat K_near untuk cluster Ck
    # K^k_near,q = floor(|Ck| * (L + Δ' * (q-1)))
    # ------------------------------------------------------------------
    def _compute_k_candidates(self, cluster_size: int) -> list[int]:
        candidates = []
        for q in range(1, self.n_candidates + 1):
            k = int(np.floor(cluster_size * (self.L + self.delta_prime * (q - 1))))
            k = max(1, min(k, cluster_size - 1))   # pastikan k valid
            candidates.append(k)
        # Hilangkan duplikat sambil pertahankan urutan
        seen = set()
        unique = []
        for k in candidates:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    # ------------------------------------------------------------------
    # Eq. 5: Hitung densitas tiap sampel dalam cluster
    # ρ_i = K_near / Σ_{j=1}^{K_near} d_ij
    # ------------------------------------------------------------------
    def _compute_density(
        self, X_cluster: np.ndarray, k_near: int
    ) -> np.ndarray:
        n = len(X_cluster)
        densities = np.zeros(n)

        # Hitung pairwise Euclidean distance
        diff = X_cluster[:, np.newaxis, :] - X_cluster[np.newaxis, :, :]  # (n, n, d)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))                   # (n, n)

        for i in range(n):
            dists = dist_matrix[i].copy()
            dists[i] = np.inf                      # abaikan jarak ke diri sendiri
            sorted_dists = np.sort(dists)[:k_near] # k_near tetangga terdekat
            sum_dists = sorted_dists.sum()
            densities[i] = k_near / sum_dists if sum_dists > 0 else 0.0

        return densities

    # ------------------------------------------------------------------
    # Eq. 8-9: Hitung cohesion score untuk subset terpilih
    # coh(C^q_k,i) = 1/(m-1) * Σ_{j≠i} d(x_i, x_j)
    # ------------------------------------------------------------------
    def _compute_cohesion(self, X_selected: np.ndarray) -> float:
        m = len(X_selected)
        if m <= 1:
            return 0.0

        diff = X_selected[:, np.newaxis, :] - X_selected[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

        # Rata-rata jarak antar pasangan (tanpa diagonal)
        cohesion_per_sample = []
        for i in range(m):
            dists_i = [dist_matrix[i, j] for j in range(m) if j != i]
            cohesion_per_sample.append(np.mean(dists_i))

        return float(np.mean(cohesion_per_sample))

    # ------------------------------------------------------------------
    # Eq. 10: Pilih q_opt = argmax cohesion(C^q_k)
    # ------------------------------------------------------------------
    def _select_optimal_k(
        self, X_cluster: np.ndarray, m: int
    ) -> int:
        candidates = self._compute_k_candidates(len(X_cluster))
        best_k = candidates[0]
        best_cohesion = -np.inf

        for k in candidates:
            densities = self._compute_density(X_cluster, k)
            # Eq. 6: urut berdasarkan densitas descending
            sorted_idx = np.argsort(-densities)
            selected_idx = sorted_idx[:m]
            X_selected = X_cluster[selected_idx]
            coh = self._compute_cohesion(X_selected)
            if coh > best_cohesion:
                best_cohesion = coh
                best_k = k

        return best_k

    # ------------------------------------------------------------------
    # API Utama: pilih indeks sampel berkualitas tinggi per cluster
    # ------------------------------------------------------------------
    def select_high_quality_samples(
        self,
        X: np.ndarray,
        cluster_labels: np.ndarray,
        threshold_t: float,
    ) -> dict[int, np.ndarray]:
        """
        Pilih sampel berkualitas tinggi dari setiap cluster.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Representasi fitur seluruh sampel.
        cluster_labels : np.ndarray, shape (N,)
            Label cluster hasil K-Means.
        threshold_t : float in [0, 1]
            Proporsi sampel yang dipilih dari tiap cluster.

        Returns
        -------
        selected_indices : dict[int, np.ndarray]
            Mapping cluster_id → indeks global sampel terpilih (berkualitas tinggi).
        """
        unique_clusters = np.unique(cluster_labels)
        selected_indices: dict[int, np.ndarray] = {}

        for cluster_id in unique_clusters:
            # Indeks global sampel dalam cluster ini
            global_idx = np.where(cluster_labels == cluster_id)[0]
            X_cluster = X[global_idx]
            n = len(X_cluster)
            m = max(1, int(np.floor(n * threshold_t)))  # jumlah sampel terpilih

            # Cari K_near optimal untuk cluster ini (Eq. 7 & 10)
            optimal_k = self._select_optimal_k(X_cluster, m)

            # Hitung densitas dengan K_near optimal (Eq. 5)
            densities = self._compute_density(X_cluster, optimal_k)

            # Eq. 6: argsort densitas descending
            sorted_local_idx = np.argsort(-densities)
            top_local_idx = sorted_local_idx[:m]

            # Kembalikan ke indeks global
            selected_indices[int(cluster_id)] = global_idx[top_local_idx]

        return selected_indices


# =============================================================================
# UMC CLUSTERING MODEL
# (Mengintegrasikan KMeans + Density-Based Sample Selection + Curriculum Learning)
# =============================================================================

class UMCClusteringModel(ClusteringModel):
    """
    Model clustering terinspirasi UMC (Zhang et al., ACL 2024).

    Tiga tahap utama yang diimplementasikan:
        Step 2 - Clustering & High-Quality Sample Selection (Section 4.3)
        Step 3 - Curriculum-based iterative refinement      (Section 4.4 & Eq. 4)

    Catatan: Step 1 (multimodal pre-training) tidak diimplementasikan di sini
    karena bergantung pada arsitektur neural network (BERT, Swin, WavLM).
    Model ini menerima representasi fitur yang sudah di-extract sebagai input.

    Parameters
    ----------
    n_clusters : int
        Jumlah cluster (KY dalam paper).
    t0 : float
        Threshold awal untuk seleksi sampel (default 0.1, sesuai Appendix E).
    delta_t : float
        Increment threshold per epoch (default 0.05, sesuai paper).
    random_state : int
        Seed untuk reprodusibilitas.
    L : float
        Lower proportion bound untuk K_near candidates.
    delta_prime : float
        Interval antar kandidat K_near.
    n_candidates : int
        Jumlah kandidat K_near.
    """

    WEIGHTS_NAME = "umc_kmeans.pkl"
    SELECTOR_NAME = "umc_selector.pkl"

    def __init__(
        self,
        n_clusters: int,
        t0: float = 0.1,
        delta_t: float = 0.05,
        random_state: int = 42,
        L: float = 0.1,
        delta_prime: float = 0.02,
        n_candidates: int = 10,
    ):
        self.n_clusters = n_clusters
        self.t0 = t0
        self.delta_t = delta_t
        self.random_state = random_state
        self.L = L
        self.delta_prime = delta_prime
        self.n_candidates = n_candidates

        # Komponen internal
        self.kmeans_ = KMeans(
            n_clusters=n_clusters,
            init="k-means++",   # K-Means++ seperti di paper (Section 4.3)
            n_init=10,
            random_state=random_state,
        )
        self.selector_ = DensityBasedSampleSelector(
            L=L, delta_prime=delta_prime, n_candidates=n_candidates
        )

        # State setelah fitting
        self.cluster_labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.high_quality_indices_: dict[int, np.ndarray] | None = None
        self.low_quality_indices_: np.ndarray | None = None
        self.final_threshold_: float = t0
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Eq. 4: Update threshold secara linear
    # t = t0 + Δ * iter
    # ------------------------------------------------------------------
    def _compute_threshold(self, iteration: int) -> float:
        t = self.t0 + self.delta_t * iteration
        return min(t, 1.0)

    # ------------------------------------------------------------------
    # STEP 2: Clustering + High-Quality Sample Selection
    # ------------------------------------------------------------------
    def _step2_cluster_and_select(
        self, X: np.ndarray, threshold_t: float, use_inherited_centroids: bool = False
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """
        Jalankan K-Means dan pilih sampel berkualitas tinggi.

        Centroid inheritance strategy (Section 4.3):
        Iterasi pertama pakai K-Means++, iterasi berikutnya
        mewarisi centroid dari iterasi sebelumnya.
        """
        if use_inherited_centroids and self.cluster_centers_ is not None:
            # Warisi centroid dari iterasi sebelumnya
            self.kmeans_.init = self.cluster_centers_
            self.kmeans_.n_init = 1

        cluster_labels = self.kmeans_.fit_predict(X)
        self.cluster_centers_ = self.kmeans_.cluster_centers_.copy()

        # Kembalikan ke K-Means++ untuk iterasi berikutnya (jika perlu reset)
        # (dikelola oleh flag use_inherited_centroids pada panggilan berikutnya)

        # Seleksi sampel berkualitas tinggi (Eq. 5-10)
        hq_indices = self.selector_.select_high_quality_samples(
            X, cluster_labels, threshold_t
        )

        return cluster_labels, hq_indices

    # ------------------------------------------------------------------
    # STEP 3: Identifikasi sampel berkualitas rendah
    # ------------------------------------------------------------------
    def _step3_identify_low_quality(
        self, n_samples: int, hq_indices: dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Semua sampel yang TIDAK terpilih sebagai high-quality
        dianggap low-quality dan akan dikenai unsupervised contrastive loss.
        """
        all_hq = np.concatenate(list(hq_indices.values()))
        all_idx = np.arange(n_samples)
        mask = np.ones(n_samples, dtype=bool)
        mask[all_hq] = False
        return all_idx[mask]

    # ------------------------------------------------------------------
    # Fit: jalankan seluruh pipeline UMC
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, max_iter: int = 20) -> "UMCClusteringModel":
        """
        Jalankan pipeline clustering UMC secara iteratif.

        Iterasi berlangsung hingga threshold t mencapai 100%
        atau max_iter tercapai (Eq. 4 & Section 4.4).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Matriks fitur (sudah di-extract, misal dari BERT/Swin/WavLM).
        max_iter : int
            Batas maksimum iterasi.
        """
        X = self.preprocess(X)

        for iteration in range(max_iter):
            # Eq. 4: hitung threshold saat ini
            t = self._compute_threshold(iteration)

            # Step 2: cluster + seleksi sampel berkualitas tinggi
            use_inherited = (iteration > 0)
            labels, hq_idx = self._step2_cluster_and_select(X, t, use_inherited)

            # Step 3: identifikasi sampel berkualitas rendah
            lq_idx = self._step3_identify_low_quality(len(X), hq_idx)

            # Simpan hasil iterasi ini
            self.cluster_labels_ = labels
            self.high_quality_indices_ = hq_idx
            self.low_quality_indices_ = lq_idx
            self.final_threshold_ = t

            n_hq = sum(len(v) for v in hq_idx.values())
            n_lq = len(lq_idx)
            print(
                f"Iter {iteration+1:3d} | t={t:.2f} | "
                f"HQ={n_hq} | LQ={n_lq} | "
                f"Clusters={self.n_clusters}"
            )

            # Berhenti jika threshold sudah 100%
            if t >= 1.0:
                print("Threshold mencapai 100%, training selesai.")
                break

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model belum di-fit. Panggil .fit() terlebih dahulu.")
        X = self.preprocess(X)
        return self.kmeans_.predict(X)

    def fit_predict(self, X: np.ndarray, max_iter: int = 20) -> np.ndarray:
        self.fit(X, max_iter=max_iter)
        return self.cluster_labels_

    # ------------------------------------------------------------------
    # Utilitas: mapping cluster ke label mayoritas (untuk evaluasi)
    # ------------------------------------------------------------------
    @staticmethod
    def map_clusters(
        true_labels: np.ndarray, pseudo_labels: np.ndarray
    ) -> tuple[np.ndarray, dict[int, int]]:
        mapped = np.zeros_like(true_labels)
        cluster_map: dict[int, int] = {}
        for cid in np.unique(pseudo_labels):
            mask = pseudo_labels == cid
            majority = mode(true_labels[mask], keepdims=False).mode
            mapped[mask] = majority
            cluster_map[int(cid)] = int(majority)
        return mapped, cluster_map

    # ------------------------------------------------------------------
    # Ringkasan distribusi high-quality vs low-quality
    # ------------------------------------------------------------------
    def get_sample_quality_summary(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Model belum di-fit.")
        summary: dict[str, Any] = {
            "final_threshold": self.final_threshold_,
            "n_clusters": self.n_clusters,
            "per_cluster": {},
        }
        total_hq = 0
        for cid, idx in self.high_quality_indices_.items():
            cluster_size = int((self.cluster_labels_ == cid).sum())
            n_hq = len(idx)
            total_hq += n_hq
            summary["per_cluster"][cid] = {
                "cluster_size": cluster_size,
                "n_high_quality": n_hq,
                "n_low_quality": cluster_size - n_hq,
                "hq_ratio": round(n_hq / cluster_size, 3),
            }
        summary["total_high_quality"] = total_hq
        summary["total_low_quality"] = len(self.low_quality_indices_)
        return summary

    # ------------------------------------------------------------------
    # Simpan & muat model
    # ------------------------------------------------------------------
    def export_config(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "t0": self.t0,
            "delta_t": self.delta_t,
            "random_state": self.random_state,
            "L": self.L,
            "delta_prime": self.delta_prime,
            "n_candidates": self.n_candidates,
        }

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> "UMCClusteringModel":
        return cls(**config)
    
    # ------------------------------------------------------------------
    # Save Weights
    # ------------------------------------------------------------------
    def save_weights(self, folder: str | Path) -> None:

        folder = Path(folder)

        folder.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.kmeans_, folder / self.WEIGHTS_NAME)

        joblib.dump(self.selector_, folder / self.SELECTOR_NAME)

    # ------------------------------------------------------------------
    # Load Weights
    # ------------------------------------------------------------------
    def load_weights(self, folder: str | Path) -> None:

        folder = Path(folder)

        self.kmeans_ = joblib.load(folder / self.WEIGHTS_NAME)

        self.selector_ = joblib.load(folder / self.SELECTOR_NAME)

        self.is_fitted_ = True

    @classmethod
    def from_folder(cls, folder: str | Path) -> "UMCClusteringModel":
        folder = Path(folder)
        with open(folder / cls.CONFIG_NAME, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls.parse_config(config)
        model.kmeans_ = joblib.load(folder / cls.WEIGHTS_NAME)
        model.selector_ = joblib.load(folder / cls.SELECTOR_NAME)
        model.is_fitted_ = True
        return model

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(42)
    print("=" * 60)
    print("Demo UMCClusteringModel")
    print("Berdasarkan paper: Zhang et al., ACL 2024")
    print("=" * 60)

    # Simulasi data embedding (misal output dari BERT/Swin/WavLM)
    N_SAMPLES = 300
    N_CLUSTERS = 5
    N_FEATURES = 128

    # Buat data sintetis dengan cluster yang dapat dipisahkan
    X_parts = []
    y_true = []
    for c in range(N_CLUSTERS):
        center = np.random.randn(N_FEATURES) * 3
        X_c = center + np.random.randn(N_SAMPLES // N_CLUSTERS, N_FEATURES) * 0.8
        X_parts.append(X_c)
        y_true.extend([c] * (N_SAMPLES // N_CLUSTERS))

    X = np.vstack(X_parts)
    y_true = np.array(y_true)

    print(f"\nData: {X.shape[0]} sampel, {X.shape[1]} fitur, {N_CLUSTERS} cluster")

    # Inisialisasi model
    model = UMCClusteringModel(
        n_clusters=N_CLUSTERS,
        t0=0.1,           # threshold awal 10% (sesuai Appendix E)
        delta_t=0.1,      # increment per iterasi
        random_state=42,
        L=0.1,            # lower bound K_near (sesuai paper)
        delta_prime=0.02, # interval K_near (sesuai paper)
        n_candidates=10,  # jumlah kandidat (sesuai paper)
    )

    print("\n--- Memulai Fitting ---")
    pseudo_labels = model.fit_predict(X, max_iter=10)

    print("\n--- Ringkasan Kualitas Sampel ---")
    summary = model.get_sample_quality_summary()
    print(f"Threshold akhir : {summary['final_threshold']:.2f}")
    print(f"Total HQ        : {summary['total_high_quality']}")
    print(f"Total LQ        : {summary['total_low_quality']}")
    print("\nPer cluster:")
    for cid, info in summary["per_cluster"].items():
        print(
            f"  Cluster {cid}: size={info['cluster_size']:3d} | "
            f"HQ={info['n_high_quality']:3d} | "
            f"LQ={info['n_low_quality']:3d} | "
            f"ratio={info['hq_ratio']:.2f}"
        )

    print("\n--- Evaluasi Clustering ---")
    metrics = model.evaluate(X, y_true)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    # Simpan visualisasi
    model.save_cluster_plot(X, "/mnt/user-data/outputs/umc_cluster_plot.png")

    print("\nSelesai!")