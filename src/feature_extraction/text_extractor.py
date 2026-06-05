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
# PREPARE TEXT FIELDS
# =========================================================

train_designations = (
    x_train["designation"]
    .fillna("")
    .astype(str)
    .tolist()
)

train_descriptions = (
    x_train["description"]
    .fillna("")
    .astype(str)
    .tolist()
)

test_designations = (
    x_test["designation"]
    .fillna("")
    .astype(str)
    .tolist()
)

test_descriptions = (
    x_test["description"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(f"Train samples: {len(train_designations)}")
print(f"Test samples : {len(test_designations)}")


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

    def __init__(self, designations, descriptions):
        self.designations = designations
        self.descriptions = descriptions

    def __len__(self):
        return len(self.designations)

    def __getitem__(self, idx):
        return (
            self.designations[idx],
            self.descriptions[idx]
        )


# =========================================================
# EXTRACT FUNCTION
# =========================================================

def extract_embeddings(
    designations,
    descriptions,
    batch_size=BATCH_SIZE,
    target_dim=2048,
    dtype=np.float16,
):
    """
    Encode designation and description separately,
    truncate using MRL,
    then concatenate.
    """

    dataset = TextDataset(
        designations,
        descriptions
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    all_embeddings = []

    with torch.inference_mode():

        for batch_idx, (batch_designation, batch_description) in enumerate(loader):

            # ----------------------------------
            # designation embedding
            # ----------------------------------

            designation_emb = model.encode(
                batch_designation,
                batch_size=batch_size,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # MRL truncation
            designation_emb = designation_emb[:, :target_dim]

            # ----------------------------------
            # description embedding
            # ----------------------------------

            description_emb = model.encode(
                batch_description,
                batch_size=batch_size,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # MRL truncation
            description_emb = description_emb[:, :target_dim]

            # ----------------------------------
            # concat
            # ----------------------------------

            embeddings = np.concatenate(
                [
                    designation_emb,
                    description_emb
                ],
                axis=1
            )

            embeddings = embeddings.astype(
                dtype,
                copy=False
            )

            all_embeddings.append(
                embeddings.copy()
            )

            del (
                designation_emb,
                description_emb,
                embeddings
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"Batch {batch_idx + 1}/{len(loader)} processed"
            )

    result = np.concatenate(
        all_embeddings,
        axis=0
    )

    del all_embeddings

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =========================================================
# TRAIN EMBEDDINGS
# =========================================================

print("\nExtracting TRAIN embeddings...")

train_embeddings = extract_embeddings(
    train_designations,
    train_descriptions,
    target_dim=2048
)

print(train_embeddings.shape)


# =========================================================
# SAVE TRAIN
# =========================================================

np.savez(
    TEXT_TRAIN_OUTPUT,

    embeddings=train_embeddings,

    labels=y_train["prdtypecode"].values,

    productid=x_train["productid"].values,

    imageid=x_train["imageid"].values,

    designation=np.array(train_designations),

    description=np.array(train_descriptions),
)

print(f"\nSaved: {TEXT_TRAIN_OUTPUT}")


# =========================================================
# TEST EMBEDDINGS
# =========================================================

print("\nExtracting TEST embeddings...")

test_embeddings = extract_embeddings(
    test_designations,
    test_descriptions,
    target_dim=2048
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

    designation=np.array(train_designations),

    description=np.array(train_descriptions),
)

print(f"\nSaved: {TEXT_TEST_OUTPUT}")


print("\nDone!")