from transforms import train_transform, val_transform
import copy
from pathlib import Path
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import LeafDiseaseDataset
from model import LeafDiseaseCNN


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
# 3. CLASS DISTRIBUTION
# --------------------------------------------------

# Get labels from training dataset
labels = [label for _, label in train_ds.samples]

# Count images in each class
class_counts = Counter(labels)

print("\nClass distribution:")

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} images")


# --------------------------------------------------
# 4. SAVE CLASS BALANCE CSV
# --------------------------------------------------

Path("reports").mkdir(exist_ok=True)

with open(
    "reports/class_balance.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "count"
    ])

    for class_id, count in sorted(class_counts.items()):
        writer.writerow([
            class_id,
            count
        ])

print(
    "\nClass distribution saved to "
    "reports/class_balance.csv"
)


# --------------------------------------------------
# 5. CALCULATE CLASS WEIGHTS
# --------------------------------------------------

# Minority classes get larger weights
class_weights = {
    class_id: 1.0 / count
    for class_id, count in class_counts.items()
}

print("\nClass weights:")

for class_id, weight in sorted(class_weights.items()):
    print(
        f"Class {class_id}: "
        f"{weight:.6f}"
    )


# --------------------------------------------------
# 6. CALCULATE SAMPLE WEIGHTS
# --------------------------------------------------

sample_weights = [
    class_weights[label]
    for label in labels
]


# --------------------------------------------------
# 7. WEIGHTED RANDOM SAMPLER
# --------------------------------------------------

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# --------------------------------------------------
# 8. DATALOADERS
# --------------------------------------------------

# IMPORTANT:
# Do not use shuffle=True when using sampler.

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 9. MODEL
# --------------------------------------------------

model = LeafDiseaseCNN().to(device)


# --------------------------------------------------
# 10. LOSS FUNCTION
# --------------------------------------------------

# We are using WeightedRandomSampler,
# so normal CrossEntropyLoss is sufficient.

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 11. OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 12. TRAINING FUNCTION
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(
        loader,
        desc="Train"
    ):

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(
            logits,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss
        total_loss += (
            loss.item() *
            images.size(0)
        )

        total_samples += images.size(0)

    return total_loss / total_samples


# --------------------------------------------------
# 13. VALIDATION FUNCTION
# --------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)

        # Validation loss
        loss = criterion(
            logits,
            labels
        )

        total_loss += (
            loss.item() *
            images.size(0)
        )

        # Predicted class
        predictions = logits.argmax(
            dim=1
        )

        # Correct predictions
        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# 14. TRAINING SETTINGS
# --------------------------------------------------

num_epochs = 25

patience = 3

best_val_loss = float("inf")

wait = 0

best_weights = None

train_losses = []

val_losses = []

val_accuracies = []


# --------------------------------------------------
# 15. CREATE FOLDERS
# --------------------------------------------------

Path("models").mkdir(
    exist_ok=True
)

Path("reports").mkdir(
    exist_ok=True
)


# --------------------------------------------------
# 16. TRAINING LOOP
# --------------------------------------------------

for epoch in range(
    1,
    num_epochs + 1
):

    print(
        f"\nEpoch {epoch}/{num_epochs}"
    )

    # -------------------------
    # Train
    # -------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # -------------------------
    # Validate
    # -------------------------

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # -------------------------
    # Store metrics
    # -------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    val_accuracies.append(
        val_acc
    )

    print(
        f"Epoch {epoch}: "
        f"train={train_loss:.4f} "
        f"val={val_loss:.4f} "
        f"acc={val_acc:.3f}"
    )


    # --------------------------------------------------
    # 17. CHECK FOR BEST MODEL
    # --------------------------------------------------

    if val_loss < best_val_loss:

        print(
            "New best validation loss!"
        )

        best_val_loss = val_loss

        wait = 0

        # Save a copy of best weights
        best_weights = copy.deepcopy(
            model.state_dict()
        )

        # Save checkpoint
        torch.save(
            model.state_dict(),
            "models/leaf_cnn_best.pth"
        )

    else:

        wait += 1

        print(
            f"No improvement. "
            f"Patience: "
            f"{wait}/{patience}"
        )


    # --------------------------------------------------
    # 18. EARLY STOPPING
    # --------------------------------------------------

    if wait >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# --------------------------------------------------
# 19. RESTORE BEST MODEL
# --------------------------------------------------

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )

    print(
        "\nBest model weights restored."
    )


# --------------------------------------------------
# 20. SAVE FINAL BEST MODEL
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "models/leaf_cnn_best.pth"
)

print(
    "Best model saved to "
    "models/leaf_cnn_best.pth"
)


# --------------------------------------------------
# 21. PLOT TRAINING AND VALIDATION LOSS
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss "
    "(Weighted Sampling)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/training_curves.png"
)

plt.close()

print(
    "Loss curve saved to "
    "reports/training_curves.png"
)


# --------------------------------------------------
# 22. SAVE TRAINING LOG
# --------------------------------------------------

with open(
    "reports/training_log.txt",
    "w"
) as f:

    f.write(
        "Plant Leaf Disease Predictor\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        "Class imbalance handling:\n"
    )

    f.write(
        "WeightedRandomSampler\n\n"
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
        f"Best validation loss: "
        f"{best_val_loss:.4f}\n"
    )

    f.write(
        f"Epochs completed: "
        f"{len(train_losses)}\n"
    )

    if val_accuracies:

        best_accuracy = max(
            val_accuracies
        )

        f.write(
            f"Best validation accuracy "
            f"after balancing: "
            f"{best_accuracy:.4f}\n"
        )

    f.write(
        "\nClass distribution:\n"
    )

    for class_id, count in sorted(
        class_counts.items()
    ):

        f.write(
            f"Class {class_id}: "
            f"{count}\n"
        )

    f.write(
        "\nClass weights:\n"
    )

    for class_id, weight in sorted(
        class_weights.items()
    ):

        f.write(
            f"Class {class_id}: "
            f"{weight:.6f}\n"
        )

    f.write(
        "\nDay 7 baseline validation "
        "accuracy before balancing: "
        "0.8725\n"
    )

    if val_accuracies:

        f.write(
            f"Day 8 validation accuracy "
            f"after balancing: "
            f"{max(val_accuracies):.4f}\n"
        )


print(
    "Training log saved to "
    "reports/training_log.txt"
)

print("\nTraining complete!")