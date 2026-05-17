import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "clustered_multimodal_embeddings.npz"

MODEL_OUTPUT = "pseudo_label_classifier.pth"

BATCH_SIZE = 64

EPOCHS = 20

LR = 1e-3

RANDOM_STATE = 42


# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

X = data["embeddings"]

y = data["pseudo_labels"]

print("\nEmbedding shape:")
print(X.shape)

print("\nPseudo label shape:")
print(y.shape)


# =========================================================
# TRAIN VALID SPLIT
# =========================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain shape:")
print(X_train.shape)

print("\nValid shape:")
print(X_valid.shape)


# =========================================================
# DATASET
# =========================================================

class EmbeddingDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# =========================================================
# DATALOADER
# =========================================================

train_dataset = EmbeddingDataset(
    X_train,
    y_train
)

valid_dataset = EmbeddingDataset(
    X_valid,
    y_valid
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# MODEL
# =========================================================

class PseudoLabelClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 512),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.model(x)


# =========================================================
# INIT MODEL
# =========================================================

input_dim = X.shape[1]

num_classes = len(
    np.unique(y)
)

print(f"\nInput dimension : {input_dim}")

print(f"Number of classes : {num_classes}")


model = PseudoLabelClassifier(
    input_dim,
    num_classes
).to(device)


# =========================================================
# LOSS & OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# =========================================================
# TRAINING
# =========================================================

best_acc = 0

for epoch in range(EPOCHS):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    preds = []

    targets = []

    with torch.no_grad():

        for X_batch, y_batch in valid_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            pred = outputs.argmax(dim=1)

            preds.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                y_batch.cpu().numpy()
            )

    acc = accuracy_score(
        targets,
        preds
    )

    f1 = f1_score(
        targets,
        preds,
        average="weighted"
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
        )

        print(
            f"Best model saved "
            f"(Acc={best_acc:.4f})"
        )


# =========================================================
# FINAL EVALUATION
# =========================================================

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

print(
    classification_report(
        targets,
        preds
    )
)

cm = confusion_matrix(
    targets,
    preds
)

print("\nConfusion Matrix:")

print(cm)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

prediction_df = {

    "true_label": targets,

    "pred_label": preds
}

import pandas as pd

prediction_df = pd.DataFrame(
    prediction_df
)

prediction_df.to_csv(
    "pseudo_label_predictions.csv",
    index=False
)

print("\nPrediction CSV saved")


# =========================================================
# DONE
# =========================================================

print("\nDone!")