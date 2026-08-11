from transforms import train_transform,val_transform
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LeafDiseaseDataset, transform
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


print("Training images:", len(train_ds))
print("Validation images:", len(val_ds))


# --------------------------------------------------
# 4. MODEL
# --------------------------------------------------

model = LeafDiseaseCNN().to(device)


# --------------------------------------------------
# 5. LOSS FUNCTION
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 6. OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 7. TRAINING FUNCTION
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
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


# --------------------------------------------------
# 8. VALIDATION FUNCTION
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
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        # Predicted class
        predictions = logits.argmax(dim=1)

        # Number of correct predictions
        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# 9. TRAINING SETTINGS
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
# 10. CREATE MODEL FOLDER
# --------------------------------------------------

Path("models").mkdir(
    exist_ok=True
)

Path("reports").mkdir(
    exist_ok=True
)


# --------------------------------------------------
# 11. TRAINING LOOP
# --------------------------------------------------

for epoch in range(1, num_epochs + 1):

    print(
        f"\nEpoch {epoch}/{num_epochs}"
    )

    # Train
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # Validate
    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # Store metrics
    train_losses.append(train_loss)

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch}: "
        f"train={train_loss:.4f} "
        f"val={val_loss:.4f} "
        f"acc={val_acc:.3f}"
    )


    # --------------------------------------------------
    # 12. CHECK FOR BEST MODEL
    # --------------------------------------------------

    if val_loss < best_val_loss:

        print("New best validation loss!")

        best_val_loss = val_loss

        wait = 0

        # Save a copy of the best weights
        best_weights = copy.deepcopy(
            model.state_dict()
        )

        # Save best checkpoint
        torch.save(
            model.state_dict(),
            "models/leaf_cnn_best.pth"
        )

    else:

        wait += 1

        print(
            f"No improvement. "
            f"Patience: {wait}/{patience}"
        )


    # --------------------------------------------------
    # 13. EARLY STOPPING
    # --------------------------------------------------

    if wait >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# --------------------------------------------------
# 14. RESTORE BEST MODEL
# --------------------------------------------------

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )

    print(
        "\nBest model weights restored."
    )


# --------------------------------------------------
# 15. SAVE FINAL BEST MODEL
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
# 16. PLOT TRAINING AND VALIDATION LOSS
# --------------------------------------------------

plt.figure(figsize=(8, 5))

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
    "Training and Validation Loss"
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
# 17. SAVE TRAINING LOG
# --------------------------------------------------

with open(
    "reports/training_log.txt",
    "w"
) as f:

    f.write("Plant Leaf Disease Predictor\n")
    f.write("============================\n\n")

    f.write(
        f"Device: {device}\n"
    )

    f.write(
        f"Training images: {len(train_ds)}\n"
    )

    f.write(
        f"Validation images: {len(val_ds)}\n"
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
        f.write(
            f"Best validation accuracy: "
            f"{max(val_accuracies):.4f}\n"
        )


print(
    "Training log saved to "
    "reports/training_log.txt"
)

print("\nTraining complete!")