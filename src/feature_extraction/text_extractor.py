import numpy as np
import pandas as pd
import torch
import gc
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from src.config import *

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================================================
# LOAD DATA
# =========================================================

x_train = pd.read_csv(X_TEXT_TRAIN_PATH)
y_train = pd.read_csv(Y_TEXT_TRAIN_PATH)

x_test = pd.read_csv(X_TEXT_TEST_PATH)


# =========================================================
# COMBINE TEXT
# =========================================================

def combine_text(df):

    designation = (
        df["designation"]
        .fillna("")
        .astype(str)
    )

    description = (
        df["description"]
        .fillna("")
        .astype(str)
    )

    texts = (
        designation + " " + description
    )

    return texts.tolist()


train_texts = combine_text(x_train)
test_texts  = combine_text(x_test)

print(f"Train texts: {len(train_texts)}")
print(f"Test texts : {len(test_texts)}")


# =========================================================
# LOAD MODEL
# =========================================================

model = SentenceTransformer(
    TEXT_MODEL_NAME,
    trust_remote_code=True,
    device=device
)

print("Model loaded")


# =========================================================
# DATASET
# =========================================================

class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# =========================================================
# EXTRACT FUNCTION
# =========================================================

def extract_embeddings(
    texts,
    batch_size=BATCH_SIZE,          # kecilkan untuk model 8B
    dtype=np.float16,
):
    """
    VRAM-optimized embedding extraction.
    """

    dataset = TextDataset(texts)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    all_embeddings = []

    # inference_mode > no_grad
    with torch.inference_mode():

        for batch_idx, batch_texts in enumerate(loader):

            # =========================
            # EMBEDDING
            # =========================

            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,

                # IMPORTANT
                convert_to_numpy=True,
                convert_to_tensor=False,

                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # force compact dtype
            embeddings = embeddings.astype(dtype, copy=False)

            # immediately move out from temporary refs
            all_embeddings.append(embeddings.copy())

            # =========================
            # CLEANUP
            # =========================

            del embeddings

            gc.collect()
            torch.cuda.empty_cache()

            print(
                f"Batch {batch_idx + 1}/{len(loader)} processed"
            )

    # =========================
    # FINAL CONCAT
    # =========================

    result = np.concatenate(
        all_embeddings,
        axis=0
    )

    # release temp lists
    del all_embeddings

    gc.collect()
    torch.cuda.empty_cache()

    return result


# =========================================================
# TRAIN EMBEDDINGS
# =========================================================

# print("\nExtracting TRAIN embeddings...")

# train_embeddings = extract_embeddings(
#     train_texts
# )

# print(train_embeddings.shape)


# =========================================================
# SAVE TRAIN
# =========================================================

# np.savez(
#     TEXT_TRAIN_OUTPUT,

#     embeddings=train_embeddings,

#     labels=y_train["prdtypecode"].values,

#     productid=x_train["productid"].values,

#     imageid=x_train["imageid"].values,

#     texts=np.array(train_texts)
# )

# print(f"\nSaved: {TEXT_TRAIN_OUTPUT}")


# =========================================================
# TEST EMBEDDINGS
# =========================================================

print("\nExtracting TEST embeddings...")

test_embeddings = extract_embeddings(
    test_texts
)

print(test_embeddings.shape)


# =========================================================
# SAVE TEST
# =========================================================

np.savez(
    TEXT_TEST_OUTPUT,

    embeddings=test_embeddings,

    productid=x_test["productid"].values,

    imageid=x_test["imageid"].values,

    texts=np.array(test_texts)
)

print(f"\nSaved: {TEXT_TEST_OUTPUT}")


print("\nDone!")