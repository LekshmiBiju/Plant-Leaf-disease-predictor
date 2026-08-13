import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import LeafDiseaseDataset, CLASS_NAMES
from transforms import train_transform, val_transform


# =========================================================
# 1. SETTINGS
# =========================================================

BASELINE_ACCURACY = 0.8725

BATCH_SIZE = 32

# Phase 1: classifier/head training
PHASE1_EPOCHS = 5

# Phase 2: layer4 + classifier fine-tuning
PHASE2_EPOCHS = 10

PATIENCE = 3

HEAD_LR = 1e-3
LAYER4_LR = 1e-5


# =========================================================
# 2. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 3. CREATE DIRECTORIES
# =========================================================

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


# =========================================================
# 4. DATASETS
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
# 5. DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# 6. LOAD PRETRAINED RESNET18
# =========================================================

weights = models.ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(
    weights=weights
)


# =========================================================
# 7. REPLACE FINAL CLASSIFIER
# =========================================================

num_classes = len(CLASS_NAMES)

in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    num_classes
)


# =========================================================
# 8. FREEZE COMPLETE BACKBONE
# =========================================================

for param in model.parameters():
    param.requires_grad = False


# Make classifier trainable
for param in model.fc.parameters():
    param.requires_grad = True


model = model.to(device)


# =========================================================
# 9. LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 10. VALIDATION FUNCTION
# =========================================================

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

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

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy


# =========================================================
# 11. TRAINING FUNCTION
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total = 0

    for images, labels in tqdm(
        loader,
        desc="Training"
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
# 12. TRACK TRAINING TIME
# =========================================================

start_time = time.perf_counter()


# =========================================================
# 13. BEST MODEL VARIABLES
# =========================================================

best_val_accuracy = 0.0

best_epoch = 0

best_phase = ""

best_weights = None

wait = 0

history = []


# =========================================================
# 14. PHASE 1
# FREEZE BACKBONE - TRAIN CLASSIFIER
# =========================================================

print("\n========================================")
print("PHASE 1: FROZEN BACKBONE")
print("Training classifier only")
print("========================================")


optimizer = optim.Adam(
    model.fc.parameters(),
    lr=HEAD_LR
)


for epoch in range(
    1,
    PHASE1_EPOCHS + 1
):

    print(
        f"\nPhase 1 - Epoch "
        f"{epoch}/{PHASE1_EPOCHS}"
    )

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
        device
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

    history.append({
        "phase": "frozen_backbone",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch

        best_phase = "phase1"

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        wait = 0

        print("New best model!")

    else:

        wait += 1


# =========================================================
# 15. PHASE 2
# UNFREEZE LAYER4
# =========================================================

print("\n========================================")
print("PHASE 2: FINE-TUNING")
print("Unfreezing layer4 + classifier")
print("========================================")


# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True


# Create parameter groups
optimizer = optim.Adam(
    [
        {
            "params": model.fc.parameters(),
            "lr": HEAD_LR
        },
        {
            "params": model.layer4.parameters(),
            "lr": LAYER4_LR
        }
    ]
)


wait = 0


for epoch in range(
    1,
    PHASE2_EPOCHS + 1
):

    print(
        f"\nPhase 2 - Epoch "
        f"{epoch}/{PHASE2_EPOCHS}"
    )

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
        device
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

    history.append({
        "phase": "fine_tuning_layer4",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch

        best_phase = "phase2"

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        wait = 0

        print("New best model!")

    else:

        wait += 1

        print(
            f"No improvement: "
            f"{wait}/{PATIENCE}"
        )

    if wait >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# 16. RESTORE BEST MODEL
# =========================================================

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )

    print(
        "\nBest model weights restored."
    )


# =========================================================
# 17. TRAINING TIME
# =========================================================

end_time = time.perf_counter()

training_time = (
    end_time - start_time
)

training_minutes = (
    training_time / 60
)


# =========================================================
# 18. SAVE CLASS NAMES
# =========================================================

with open(
    "models/class_names.json",
    "w"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# =========================================================
# 19. SAVE BEST CHECKPOINT
# =========================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "epoch":
        best_epoch,

    "best_validation_accuracy":
        best_val_accuracy,

    "baseline_accuracy":
        BASELINE_ACCURACY,

    "meets_baseline":
        best_val_accuracy > BASELINE_ACCURACY,

    "class_names":
        CLASS_NAMES,

    "best_phase":
        best_phase,

    "device":
        str(device),

    "training_time_seconds":
        training_time
}


torch.save(
    checkpoint,
    "models/resnet18_leaf_best.pth"
)


# =========================================================
# 20. SAVE TRAINING LOG
# =========================================================

with open(
    "reports/resnet18_finetuning_log.txt",
    "w"
) as f:

    f.write(
        "ResNet18 Fine-Tuning Report\n"
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
        f"{len(val_ds)}\n\n"
    )

    f.write(
        "Phase 1: Frozen backbone, "
        "classifier training\n"
    )

    f.write(
        f"Phase 1 epochs: "
        f"{PHASE1_EPOCHS}\n"
    )

    f.write(
        f"Phase 1 learning rate: "
        f"{HEAD_LR}\n\n"
    )

    f.write(
        "Phase 2: Unfreeze layer4 "
        "and fine-tune\n"
    )

    f.write(
        f"Phase 2 epochs: "
        f"{PHASE2_EPOCHS}\n"
    )

    f.write(
        f"Layer4 learning rate: "
        f"{LAYER4_LR}\n\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}\n"
    )

    f.write(
        f"Baseline CNN accuracy: "
        f"{BASELINE_ACCURACY:.4f}\n"
    )

    f.write(
        f"Meets baseline: "
        f"{best_val_accuracy > BASELINE_ACCURACY}\n"
    )

    f.write(
        f"Best phase: "
        f"{best_phase}\n"
    )

    f.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Training time: "
        f"{training_minutes:.2f} minutes\n"
    )


# =========================================================
# 21. FINAL OUTPUT
# =========================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Baseline CNN accuracy: "
    f"{BASELINE_ACCURACY:.4f}"
)

print(
    "Meets baseline:",
    best_val_accuracy > BASELINE_ACCURACY
)

print(
    f"Training time: "
    f"{training_minutes:.2f} minutes"
)

print(
    "\nBest checkpoint saved to:"
)

print(
    "models/resnet18_leaf_best.pth"
)

print(
    "\nClass names saved to:"
)

print(
    "models/class_names.json"
)

print(
    "\nTraining log saved to:"
)

print(
    "reports/resnet18_finetuning_log.txt"
)