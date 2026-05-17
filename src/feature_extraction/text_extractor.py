import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIG
# =========================================================

X_TRAIN_PATH = "C:/Personal/coding/S2/sem2/pm/Dataset/X_train_update.csv"
Y_TRAIN_PATH = "C:/Personal/coding/S2/sem2/pm/Dataset/Y_train_CVw08PX.csv"
X_TEST_PATH  = "C:/Personal/coding/S2/sem2/pm/Dataset/X_test_update.csv"

TRAIN_OUTPUT = "C:/Personal/coding/S2/sem2/pm/Dataset/features/text/train_text_embeddings.npz"
TEST_OUTPUT  = "C:/Personal/coding/S2/sem2/pm/Dataset/features/text/test_text_embeddings.npz"

MODEL_NAME = "nvidia/llama-embed-nemotron-8b"

BATCH_SIZE = 16


# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================================================
# LOAD DATA
# =========================================================

x_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)

x_test = pd.read_csv(X_TEST_PATH)


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
    MODEL_NAME,
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

def extract_embeddings(texts):

    dataset = TextDataset(texts)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_embeddings = []

    with torch.no_grad():

        for batch_idx, batch_texts in enumerate(loader):

            embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            embeddings = embeddings.astype(np.float16)

            all_embeddings.append(embeddings)

            print(
                f"Batch {batch_idx + 1}/{len(loader)} processed"
            )

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    return all_embeddings


# =========================================================
# TRAIN EMBEDDINGS
# =========================================================

print("\nExtracting TRAIN embeddings...")

train_embeddings = extract_embeddings(
    train_texts
)

print(train_embeddings.shape)


# =========================================================
# SAVE TRAIN
# =========================================================

np.savez(
    TRAIN_OUTPUT,

    embeddings=train_embeddings,

    labels=y_train["prdtypecode"].values,

    productid=x_train["productid"].values,

    imageid=x_train["imageid"].values,

    texts=np.array(train_texts)
)

print(f"\nSaved: {TRAIN_OUTPUT}")


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
    TEST_OUTPUT,

    embeddings=test_embeddings,

    productid=x_test["productid"].values,

    imageid=x_test["imageid"].values,

    texts=np.array(test_texts)
)

print(f"\nSaved: {TEST_OUTPUT}")


print("\nDone!")