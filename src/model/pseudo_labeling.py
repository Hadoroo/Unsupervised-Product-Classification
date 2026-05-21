import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os

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

from src.config import *

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    MULTIMODAL_CLUSTER_OUTPUT,
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

model_exists = os.path.exists(MODEL_OUTPUT)

if model_exists:

    print("\nExisting model found.")

    model.load_state_dict(
        torch.load(
            MODEL_OUTPUT,
            map_location=device
        )
    )

    print("Model loaded. Skipping training.")

else:

    print("\nNo existing model found.")
    print("Training will start.")


# =========================================================
# LOSS & OPTIMIZER
# =========================================================
if not model_exists:
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
def evaluate_model():

    model.eval()

    preds = []

    targets = []

    with torch.inference_mode():

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

    return targets, preds

targets, preds = evaluate_model()

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

report_text = classification_report(
    targets,
    preds
)

print(report_text)

report_dict = classification_report(
    targets,
    preds,
    output_dict=True
)

report_df = pd.DataFrame(
    report_dict
).transpose()

markdown_report = report_df.to_markdown()

with open(MD_OUTPUT, "w", encoding="utf-8") as f:

    f.write("# Classification Report\n\n")

    f.write(markdown_report)

print(f"\nMarkdown report saved: {MD_OUTPUT}")

cm = confusion_matrix(
    targets,
    preds
)

print("\nConfusion Matrix:")

print(cm)

# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(cm)

# ==========================================
# LABELS
# ==========================================

classes = np.unique(targets)

ax.set_xticks(np.arange(len(classes)))

ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes)

ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Label")

ax.set_ylabel("True Label")

ax.set_title("Confusion Matrix")

# ==========================================
# WRITE VALUES INSIDE CELLS
# ==========================================

threshold = cm.max() / 2

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        value = cm[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="black" if value > threshold else "white"
        )

# ==========================================
# COLORBAR
# ==========================================

fig.colorbar(im)

plt.tight_layout()

plt.savefig(
    PNG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix saved: {PNG_OUTPUT}")


# =========================================================
# SAVE PREDICTIONS
# =========================================================

prediction_df = {

    "true_label": targets,

    "pred_label": preds
}

prediction_df = pd.DataFrame(
    prediction_df
)

prediction_df.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

print("\nPrediction CSV saved")


# =========================================================
# DONE
# =========================================================

print("\nDone!")