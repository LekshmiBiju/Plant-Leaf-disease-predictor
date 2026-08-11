from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from dataset import LeafDiseaseDataset
from transforms import train_transform, val_transform


# =========================================================
# 1. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 2. DATASETS
# =========================================================

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


# =========================================================
# 3. DATALOADERS
# =========================================================

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


# =========================================================
# 4. LOAD PRETRAINED RESNET18
# =========================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)


# =========================================================
# 5. FREEZE BACKBONE
# =========================================================

for param in model.parameters():
    param.requires_grad = False


# Replace final classifier
model.fc = nn.Linear(
    model.fc.in_features,
    4
)


model = model.to(device)


# =========================================================
# 6. LOSS AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-3
)


# =========================================================
# 7. TRAINING FUNCTION
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total = 0

    for images, labels in tqdm(
        train_loader,
        desc="Train"
    ):

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

        total += images.size(0)

    return total_loss / total


# =========================================================
# 8. VALIDATION FUNCTION
# =========================================================

@torch.no_grad()
def validate():

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    loss = total_loss / total

    return loss, accuracy


# =========================================================
# 9. TRAINING SETTINGS
# =========================================================

num_epochs = 10

best_accuracy = 0.0

best_weights = None

accuracies = []


# =========================================================
# 10. CREATE FOLDERS
# =========================================================

Path("models").mkdir(
    exist_ok=True
)

Path("reports").mkdir(
    exist_ok=True
)


# =========================================================
# 11. PHASE 1 — TRAIN CLASSIFIER
# =========================================================

print("\n================================")
print("PHASE 1: FROZEN RESNET18")
print("================================")


for epoch in range(1, num_epochs + 1):

    print(
        f"\nEpoch {epoch}/{num_epochs}"
    )

    train_loss = train_one_epoch()

    val_loss, val_accuracy = validate()

    accuracies.append(
        val_accuracy
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            "models/resnet18_best.pth"
        )

        print("New best model saved!")


# =========================================================
# 12. PHASE 2 — UNFREEZE LAYER4
# =========================================================

print("\n================================")
print("PHASE 2: UNFREEZE LAYER4")
print("================================")


for param in model.layer4.parameters():
    param.requires_grad = True


optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4
)


# =========================================================
# 13. FINE-TUNING
# =========================================================

for epoch in range(1, 6):

    print(
        f"\nFine-tuning Epoch "
        f"{epoch}/5"
    )

    train_loss = train_one_epoch()

    val_loss, val_accuracy = validate()

    accuracies.append(
        val_accuracy
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            "models/resnet18_best.pth"
        )

        print("New best model saved!")


# =========================================================
# 14. RESTORE BEST MODEL
# =========================================================

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )


# =========================================================
# 15. FINAL TEST
# =========================================================

final_loss, final_accuracy = validate()

print("\n================================")
print("FINAL RESNET18 RESULTS")
print("================================")

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Final validation accuracy: "
    f"{final_accuracy:.4f}"
)

print(
    "Best model saved to "
    "models/resnet18_best.pth"
)


# =========================================================
# 16. SAVE METRICS REPORT
# =========================================================

with open(
    "reports/resnet18_metrics.txt",
    "w"
) as f:

    f.write(
        "ResNet18 Transfer Learning\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    f.write(
        f"Training images: "
        f"{len(train_ds)}\n"
    )

    f.write(
        f"Validation images: "
        f"{len(val_ds)}\n"
    )

    f.write(
        "Strategy: Freeze backbone, "
        "train classifier, then unfreeze "
        "layer4 for fine-tuning.\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{best_accuracy:.4f}\n"
    )

    f.write(
        f"Baseline CNN accuracy: "
        f"0.8725\n"
    )

    f.write(
        f"Meets baseline: "
        f"{best_accuracy >= 0.8725}\n"
    )

    f.write(
        "\nBest model: "
        "models/resnet18_best.pth\n"
    )


print(
    "\nMetrics saved to "
    "reports/resnet18_metrics.txt"
)

print("\nTraining complete!")