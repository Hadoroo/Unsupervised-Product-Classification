import numpy as np
from src.config import TRAIN_MULTIMODAL_OUTPUT, UNIMODAL_MULTIMODAL_OUTPUT
def split_unimodal_from_multimodal(multimodal_path, output_path):
    """
    Split existing 6144-D multimodal embeddings back to unimodal.
    """
    data = np.load(multimodal_path, allow_pickle=True)
    X_mm = data["embeddings"].astype(np.float64)  # (N, 6144)
    
    # Split
    z_text_norm = X_mm[:, :4096].astype(np.float32)   # 4096-D
    z_image_norm = X_mm[:, 4096:].astype(np.float32)  # 2048-D
    
    # Further split text
    z_text_designation = z_text_norm[:, :2048]
    z_text_description = z_text_norm[:, 2048:]
    
    # Save dengan SEMUA key yang dibutuhkan
    np.savez(
        output_path,
        # Unimodal
        z_image_norm=z_image_norm,
        z_text_designation=z_text_designation,
        z_text_description=z_text_description,
        
        # Text concat (dua nama untuk kompatibilitas)
        z_text_norm=z_text_norm,      # nama lama / fallback
        z_text_concat=z_text_norm,    # nama baru / ideal
        
        # Multimodal
        z_multimodal=X_mm.astype(np.float32),
        
        # Metadata
        labels=data["labels"],
        productids=data["productid"],
        imageids=data["imageid"],
        designation=data["designation"],
        description=data["description"]
    )
    
    return output_path

if __name__ == "__main__":
    split_unimodal_from_multimodal(TRAIN_MULTIMODAL_OUTPUT, UNIMODAL_MULTIMODAL_OUTPUT)