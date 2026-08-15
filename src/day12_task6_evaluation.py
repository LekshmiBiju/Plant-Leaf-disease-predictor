from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

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
# 2. CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]


# =========================================================
# 3. TEST DATASET
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
# 4. LOAD MODEL
# =========================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        "models/resnet18_day12_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# 5. PREDICTIONS
# =========================================================

y_true = []
y_pred = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu().tolist()

        y_pred.extend(predictions)

        y_true.extend(
            labels.tolist()
        )


# =========================================================
# 6. CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(report)


# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


# =========================================================
# 8. SAVE REPORT
# =========================================================

Path("reports").mkdir(
    exist_ok=True
)

with open(
    "reports/task6_classification_report.md",
    "w"
) as f:

    f.write(
        "# Task 6 Classification Report\n\n"
    )

    f.write(
        "## Model\n\n"
    )

    f.write(
        "`models/resnet18_day12_best.pth`\n\n"
    )

    f.write(
        "## Test Set\n\n"
    )

    f.write(
        f"Test images: {len(test_ds)}\n\n"
    )

    f.write(
        "## Precision, Recall and F1-Score\n\n"
    )

    f.write(
        "```text\n"
    )

    f.write(report)

    f.write(
        "```\n\n"
    )

    f.write(
        "## Confusion Matrix\n\n"
    )

    f.write(
        "```text\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n```\n"
    )


print(
    "\nClassification report saved to "
    "reports/task6_classification_report.md"
)