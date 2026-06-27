# ============================================
# FILE: experiments/ablation_unimodal.py
# ============================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.config import *
from src.model.base_model import ClusteringModel
from src.model.kmeans import KMeansClustering

def run_unimodal_ablation(
    data_path: Path,
    output_root: Path | None = None
) -> dict[str, dict]:
    """
    Run complete unimodal ablation study.
    
    Variants:
    1. Image Only (ResNet50, 2048-D)
    2. Text Designation Only (Qwen3, 2048-D)
    3. Text Description Only (Qwen3, 2048-D)
    4. Text Concat (Designation + Description, 4096-D)
    5. Multimodal (Image + Text, 6144-D) [baseline]
    """
    
    print("=" * 60)
    print("UNIMODAL ABLATION STUDY")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    # Define embedding variants
    EMBEDDING_VARIANTS = {
        "image_only": {
            "embedding": data.get("z_image", data.get("z_image_norm")),
            "dim": 2048,
            "description": "ResNet50 image embeddings only"
        },
        "text_designation_only": {
            "embedding": data.get("z_designation", data.get("z_text_designation")),
            "dim": 2048,
            "description": "Qwen3 designation embeddings only"
        },
        "text_description_only": {
            "embedding": data.get("z_description", data.get("z_text_description")),
            "dim": 2048,
            "description": "Qwen3 description embeddings only"
        },
        "text_concat": {
            "embedding": data.get("z_text_concat", data.get("z_text_norm")),
            "dim": 4096,
            "description": "Concatenated designation + description"
        },
        "multimodal": {
            "embedding": data.get("z_multimodal", data.get("embeddings")),
            "dim": 6144,
            "description": "Full multimodal (image + text concat)"
        }
    }
    
    # Metadata
    true_labels = data["labels"]
    productids = data.get("productid", data.get("productids"))
    imageids = data.get("imageid", data.get("imageids"))
    designation = data["designation"]
    description = data["description"]
    
    num_clusters = len(np.unique(true_labels))
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root is None:
        output_root = Path.cwd() / "outputs" / f"ablation_unimodal_{timestamp}"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}")
    
    # Results storage
    all_results = {}
    summary_rows = []
    
    # Run ablation for each variant
    for variant_name, variant_info in EMBEDDING_VARIANTS.items():
        
        print(f"\n{'='*50}")
        print(f"VARIANT: {variant_name}")
        print(f"Description: {variant_info['description']}")
        print(f"Dimensions: {variant_info['dim']}")
        print(f"{'='*50}")
        
        X = variant_info["embedding"].astype(np.float64)
        
        # Preprocess
        X = ClusteringModel.preprocess(X)
        
        # Create variant output folder
        variant_folder = output_root / variant_name
        variant_folder.mkdir(exist_ok=True)
        
        models_folder = variant_folder / "models"
        results_folder = variant_folder / "results"
        models_folder.mkdir(exist_ok=True)
        results_folder.mkdir(exist_ok=True)
        
        # Initialize model (K-Means sebagai baseline konsisten)
        model = KMeansClustering(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10
        )
        
        # Train / Predict
        print(f"Training K-Means on {variant_name}...")
        pseudo_labels = model.fit_predict(X)
        
        # Map clusters to true labels
        print(f"Mapping clusters...")
        mapped_labels, cluster_map = model.map_clusters(true_labels, pseudo_labels)
        
        # Evaluate
        print(f"Evaluating...")
        metrics = model.evaluate(X, true_labels, pseudo_labels, mapped_labels)
        
        # Print metrics
        print(f"\n--- Metrics for {variant_name} ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Save results
        # 1. Cluster assignments CSV
        df = pd.DataFrame({
            "productid": productids,
            "imageid": imageids,
            "designation": designation,
            "description": description,
            "true_label": true_labels,
            "pseudo_label": pseudo_labels,
            "mapped_label": mapped_labels
        })
        df.to_csv(results_folder / "cluster_result.csv", index=False)
        
        # 2. Metrics CSV
        metrics_df = pd.DataFrame({
            "metric": list(metrics.keys()),
            "value": list(metrics.values())
        })
        metrics_df.to_csv(results_folder / "metrics.csv", index=False)
        
        # 3. Cluster plot
        model.save_cluster_plot(X, pseudo_labels, results_folder / "cluster_plot.png")
        
        # 4. Save model
        model.save_folder(models_folder)
        
        # Store for summary
        all_results[variant_name] = {
            "metrics": metrics,
            "model": model,
            "cluster_map": cluster_map
        }
        
        summary_rows.append({
            "variant": variant_name,
            "dim": variant_info["dim"],
            "description": variant_info["description"],
            **metrics
        })
    
    # Generate summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY TABLE")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "ablation_summary.csv", index=False)
    
    # Pretty print
    print("\n" + summary_df.to_string(index=False))
    
    # Print improvement untuk SEMUA metrics yang ada
    print(f"\n{'='*60}")
    print("MULTIMODAL IMPROVEMENT ANALYSIS")
    print(f"{'='*60}")
    
    unimodal_variants = ["image_only", "text_designation_only", 
                         "text_description_only", "text_concat"]
    
    # Metrics di mana HIGHER is better
    higher_better = ["silhouette_score", "calinski_harabasz_score", 
                     "ari_score", "nmi_score", "kappa_score"]
    
    # Metrics di mana LOWER is better
    lower_better = ["davies_bouldin_score"]
    
    for metric in summary_df.columns:
        if metric in ("variant", "dim", "description"):
            continue
            
        unimodal_vals = summary_df[summary_df["variant"].isin(unimodal_variants)][metric]
        if len(unimodal_vals) == 0:
            continue
            
        best_unimodal = unimodal_vals.max()
        multimodal_val = summary_df[summary_df["variant"] == "multimodal"][metric].iloc[0]
        
        # Tentukan arah improvement
        if metric in lower_better:
            # Lower is better
            improvement = ((best_unimodal - multimodal_val) / abs(best_unimodal)) * 100
            better = multimodal_val < best_unimodal
        else:
            # Higher is better (default)
            improvement = ((multimodal_val - best_unimodal) / abs(best_unimodal)) * 100
            better = multimodal_val > best_unimodal
            
        status = "✅ BETTER" if better else "❌ WORSE"
        
        print(f"\n  {metric}:")
        print(f"    Best Unimodal: {best_unimodal:.4f}")
        print(f"    Multimodal:    {multimodal_val:.4f}")
        print(f"    Change:        {improvement:+.2f}%  {status}")
    
    return all_results


# ============================================
# RUN SCRIPT
# ============================================

if __name__ == "__main__":
    
    # Path ke data (ganti sesuai struktur Anda)
    DATA_PATH = Path("outputs/features/multimodal/unimodal_multimodal_embeddings.npz")
    
    # Atau jika menggunakan data existing yang di-split:
    # DATA_PATH = Path("data/embeddings/split_from_multimodal.npz")
    
    results = run_unimodal_ablation(DATA_PATH)