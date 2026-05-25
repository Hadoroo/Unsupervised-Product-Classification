import numpy as np

from src.config import *

data = np.load(
    TRAIN_MULTIMODAL_OUTPUT,
    allow_pickle=True
)

X = data["embeddings"].astype(np.float64)

true_labels = data["labels"]

print(np.unique(true_labels))