from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------
# 1. CLASS NAMES
# ---------------------------------------------------------

CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]

CLASS_TO_IDX = {
    name: i for i, name in enumerate(CLASS_NAMES)
}


# ---------------------------------------------------------
# 2. IMAGE TRANSFORM
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# 3. DATASET CLASS
# ---------------------------------------------------------

class LeafDiseaseDataset(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # Look inside each class folder
        for class_name in CLASS_NAMES:

            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            # Support jpg, jpeg and png images
            for img_path in class_dir.iterdir():

                if img_path.is_file() and img_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png"
                }:

                    self.samples.append(
                        (img_path, CLASS_TO_IDX[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        # Open image
        image = Image.open(path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# 4. CREATE DATASETS
# ---------------------------------------------------------

train_ds = LeafDiseaseDataset(
    "data/train",
    transform=transform
)

val_ds = LeafDiseaseDataset(
    "data/val",
    transform=transform
)


# ---------------------------------------------------------
# 5. CREATE DATALOADERS
# ---------------------------------------------------------

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------------
# 6. TEST THE DATALOADERS
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))

    images, labels = next(iter(train_loader))

    print("Train batch shape:", images.shape)
    print("Train labels:", labels[:5])

    val_images, val_labels = next(iter(val_loader))

    print("Validation batch shape:", val_images.shape)
    print("Validation labels:", val_labels[:5])