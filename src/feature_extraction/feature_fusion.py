import numpy as np
from src.config import *
# =========================================================
# LOAD IMAGE EMBEDDINGS
# =========================================================

image_data = np.load(
    IMAGE_TRAIN_OUTPUT,
    allow_pickle=True
)

image_embeddings = image_data["embeddings"]
image_filenames  = image_data["filenames"]

print("Image embeddings:")
print(image_embeddings.shape)


# =========================================================
# LOAD TEXT EMBEDDINGS
# =========================================================

text_data = np.load(
    TEXT_TRAIN_OUTPUT,
    allow_pickle=True
)

text_embeddings = text_data["embeddings"]

productids = text_data["productid"]
imageids   = text_data["imageid"]

labels = text_data["labels"]

texts = text_data["texts"]

print("\nText embeddings:")
print(text_embeddings.shape)


# =========================================================
# CHECK TOTAL DATA
# =========================================================

assert len(image_embeddings) == len(text_embeddings), \
    "Jumlah image dan text embedding berbeda!"


# =========================================================
# NORMALIZATION (RECOMMENDED)
# =========================================================

def l2_normalize(x):

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / (norm + 1e-8)


image_embeddings = l2_normalize(
    image_embeddings.astype(np.float32)
)

text_embeddings = l2_normalize(
    text_embeddings.astype(np.float32)
)


# =========================================================
# FEATURE FUSION
# =========================================================

fusion_embeddings = np.concatenate(
    [
        image_embeddings,
        text_embeddings
    ],
    axis=1
)

print("\nFusion embeddings:")
print(fusion_embeddings.shape)


# =========================================================
# SAVE
# =========================================================

np.savez(
    TRAIN_MULTIMODAL_OUTPUT,

    embeddings=fusion_embeddings.astype(np.float16),

    labels=labels,

    productid=productids,

    imageid=imageids,

    texts=texts
)

print(f"\nSaved: {TRAIN_MULTIMODAL_OUTPUT}")