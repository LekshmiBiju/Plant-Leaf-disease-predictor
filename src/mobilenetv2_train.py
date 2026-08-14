import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import LeafDiseaseDataset
from transforms import train_transform, val_transform


# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. DATASETS
# --------------------------------------------------

train_ds = LeafDiseaseDataset(
    "data/train",
    transform=train_transform
)

val_ds = LeafDiseaseDataset(
    "data/val",
    transform=val_transform
)

print("Training images:", len(train_ds))
print("Validation images:", len(val_ds))


# --------------------------------------------------
# 3. DATALOADERS
# --------------------------------------------------

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


# --------------------------------------------------
# 4. LOAD PRETRAINED MOBILENETV2
# --------------------------------------------------

weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

model = models.mobilenet_v2(
    weights=weights
)


# --------------------------------------------------
# 5. REPLACE CLASSIFIER
# --------------------------------------------------

num_classes = 4

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)


# --------------------------------------------------
# 6. FREEZE BACKBONE
# --------------------------------------------------

for param in model.features.parameters():
    param.requires_grad = False


# --------------------------------------------------
# 7. LOSS FUNCTION
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 8. OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 9. VALIDATION FUNCTION
# --------------------------------------------------

@torch.no_grad()
def validate():

    model.eval()

    correct = 0
    total = 0

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return correct / total


# --------------------------------------------------
# 10. TRAINING
# --------------------------------------------------

num_epochs = 10

best_accuracy = 0.0
best_weights = None

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


for epoch in range(1, num_epochs + 1):

    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

    val_accuracy = validate()

    avg_loss = (
        total_loss / len(train_ds)
    )

    print(
        f"Epoch {epoch}/{num_epochs} "
        f"Loss: {avg_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            "models/mobilenetv2_leaf_best.pth"
        )

        print("New best model saved.")


# --------------------------------------------------
# 11. RESTORE BEST MODEL
# --------------------------------------------------

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )


print()
print("Training complete.")
print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    "Best model saved to "
    "models/mobilenetv2_leaf_best.pth"
)