
import os
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from src.config import *

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# =========================================================
# LOAD PRETRAINED RESNET50
# =========================================================

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Remove classifier layer
model = nn.Sequential(*list(model.children())[:-1])

model = model.to(device)
model.eval()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# CUSTOM DATASET
# =========================================================

class ImageDataset(Dataset):

    def __init__(self, folder_path, transform=None):

        self.folder_path = folder_path
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        image_path = os.path.join(
            self.folder_path,
            filename
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, filename


# =========================================================
# EXTRACT EMBEDDINGS
# =========================================================

def extract_embeddings(dataset_path, output_file):

    print(f"\nProcessing: {dataset_path}")

    dataset = ImageDataset(
        folder_path=dataset_path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    all_embeddings = []
    all_filenames = []

    with torch.no_grad():

        for batch_idx, (images, filenames) in enumerate(loader):

            images = images.to(device)

            # Forward pass
            embeddings = model(images)

            # [B, 2048, 1, 1] -> [B, 2048]
            embeddings = embeddings.view(
                embeddings.size(0),
                -1
            )

            # Convert to numpy float16
            embeddings = (
                embeddings
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            # Save to RAM
            all_embeddings.append(embeddings)

            all_filenames.extend(filenames)

            print(
                f"Batch {batch_idx + 1}/{len(loader)} processed"
            )

    # =====================================================
    # COMBINE ALL BATCHES
    # =====================================================

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    all_filenames = np.array(all_filenames)

    print("\nFinal embedding shape:")
    print(all_embeddings.shape)

    # =====================================================
    # SAVE NPZ
    # =====================================================

    np.savez(
        output_file,
        embeddings=all_embeddings,
        filenames=all_filenames
    )

    print(f"\nSaved: {output_file}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # TRAIN
    extract_embeddings(
        dataset_path=IMAGE_TRAIN_PATH,
        output_file=IMAGE_TRAIN_OUTPUT
    )

    # TEST
    extract_embeddings(
        dataset_path=IMAGE_TEST_PATH,
        output_file=IMAGE_TEST_OUTPUT
    )

    print("\nDone!")