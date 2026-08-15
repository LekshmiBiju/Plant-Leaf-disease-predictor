from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from dataset import LeafDiseaseDataset
from transforms import val_transform


# =========================================================
# 1. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 2. TEST DATASET
# =========================================================

test_ds = LeafDiseaseDataset(
    "data/day12/test",
    transform=val_transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_ds))


# =========================================================
# 3. CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]


# =========================================================
# 4. LOAD RESNET18
# =========================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)


# =========================================================
# 5. LOAD BEST DAY 12 MODEL
# =========================================================

model.load_state_dict(
    torch.load(
        "models/resnet18_day12_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# 6. TEST EVALUATION
# =========================================================

correct = 0
total = 0

y_true = []
y_pred = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )


# =========================================================
# 7. CALCULATE TEST ACCURACY
# =========================================================

test_accuracy = correct / total


print("\n================================")
print("DAY 12 TEST RESULTS")
print("================================")

print(
    f"Test images: {total}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Incorrect predictions: "
    f"{total - correct}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# 8. SAVE TEST RESULTS
# =========================================================

Path("reports").mkdir(
    exist_ok=True
)

with open(
    "reports/day12_test_metrics.txt",
    "w"
) as f:

    f.write(
        "Day 12 ResNet18 Test Evaluation\n"
    )

    f.write(
        "===============================\n\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    f.write(
        f"Test images: {total}\n"
    )

    f.write(
        f"Correct predictions: {correct}\n"
    )

    f.write(
        f"Incorrect predictions: "
        f"{total - correct}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Test accuracy percentage: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        "\nModel: "
        "models/resnet18_day12_best.pth\n"
    )


# =========================================================
# 9. SAVE PREDICTIONS FOR CONFUSION MATRIX
# =========================================================

torch.save(
    {
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": CLASS_NAMES
    },
    "reports/day12_predictions.pt"
)


print(
    "\nTest metrics saved to "
    "reports/day12_test_metrics.txt"
)

print(
    "Predictions saved to "
    "reports/day12_predictions.pt"
)

print("\nTest evaluation complete!")