from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score
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

healthy_idx = CLASS_NAMES.index("healthy")


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
# 6. COLLECT PREDICTIONS AND PROBABILITIES
# =========================================================

y_true = []
y_pred = []
disease_probs = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        # Convert logits to probabilities
        probs = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        y_true.extend(
            labels.tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

        # Probability of being diseased
        disease_probability = (
            1 - probs[:, healthy_idx]
        )

        disease_probs.extend(
            disease_probability.cpu().numpy()
        )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
disease_probs = np.array(disease_probs)


# =========================================================
# 7. CLASSIFICATION REPORT
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


Path("reports").mkdir(
    exist_ok=True
)

with open(
    "reports/classification_report.txt",
    "w"
) as f:

    f.write(
        "Day 13 - ResNet18 Classification Report\n"
    )

    f.write(
        "=========================================\n\n"
    )

    f.write(report)

    f.write(
        "\nModel: models/resnet18_day12_best.pth\n"
    )

    f.write(
        f"Test images: {len(test_ds)}\n"
    )


print(
    "\nClassification report saved to "
    "reports/classification_report.txt"
)


# =========================================================
# 8. CREATE BINARY LABELS
# =========================================================
# healthy = 0
# any disease = 1


y_binary = np.array([
    0 if label == healthy_idx else 1
    for label in y_true
])


# =========================================================
# 9. PRECISION-RECALL CURVE
# =========================================================

precision, recall, thresholds = precision_recall_curve(
    y_binary,
    disease_probs
)

average_precision = average_precision_score(
    y_binary,
    disease_probs
)


# =========================================================
# 10. FIND OPERATING THRESHOLD
# =========================================================

chosen_threshold = None
chosen_precision = None
chosen_recall = None


# precision/recall have one extra value compared
# with thresholds, so use precision[:-1] and recall[:-1]

valid_indices = np.where(
    (recall[:-1] >= 0.95) &
    (precision[:-1] >= 0.80)
)[0]


if len(valid_indices) > 0:

    # Prefer highest recall.
    # If tied, prefer highest precision.

    best_index = sorted(
        valid_indices,
        key=lambda i: (
            recall[i],
            precision[i]
        ),
        reverse=True
    )[0]

    chosen_threshold = float(
        thresholds[best_index]
    )

    chosen_precision = float(
        precision[best_index]
    )

    chosen_recall = float(
        recall[best_index]
    )

else:

    # If both requirements cannot be satisfied,
    # choose the point with the highest recall
    # while maintaining precision >= 0.80.

    precision_valid = np.where(
        precision[:-1] >= 0.80
    )[0]

    if len(precision_valid) > 0:

        best_index = precision_valid[
            np.argmax(
                recall[precision_valid]
            )
        ]

        chosen_threshold = float(
            thresholds[best_index]
        )

        chosen_precision = float(
            precision[best_index]
        )

        chosen_recall = float(
            recall[best_index]
        )

    else:

        # Fallback: choose the point with
        # maximum F1 score.

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (
                precision[:-1]
                + recall[:-1]
                + 1e-8
            )
        )

        best_index = np.argmax(
            f1_scores
        )

        chosen_threshold = float(
            thresholds[best_index]
        )

        chosen_precision = float(
            precision[best_index]
        )

        chosen_recall = float(
            recall[best_index]
        )


# =========================================================
# 11. PRINT THRESHOLD RESULTS
# =========================================================

print("\n================================")
print("THRESHOLD TUNING")
print("================================")

print(
    f"Chosen threshold: "
    f"{chosen_threshold:.4f}"
)

print(
    f"Precision: "
    f"{chosen_precision:.4f}"
)

print(
    f"Recall: "
    f"{chosen_recall:.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision:.4f}"
)


# =========================================================
# 12. SAVE THRESHOLD METRICS
# =========================================================

with open(
    "reports/threshold_metrics.txt",
    "w"
) as f:

    f.write(
        " Threshold Tuning\n"
    )

    f.write(
        "=======================\n\n"
    )

    f.write(
        f"Chosen threshold: "
        f"{chosen_threshold:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{chosen_precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{chosen_recall:.4f}\n"
    )

    f.write(
        f"Average Precision: "
        f"{average_precision:.4f}\n"
    )

    f.write(
        "\nTarget: recall >= 0.95 "
        "and precision >= 0.80\n"
    )


# =========================================================
# 13. PLOT PR CURVE
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"PR curve (AP = {average_precision:.3f})"
)

plt.scatter(
    chosen_recall,
    chosen_precision,
    s=80,
    label=(
        f"Chosen threshold = "
        f"{chosen_threshold:.3f}"
    )
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Healthy vs Diseased"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/pr_curve.png",
    dpi=200
)

plt.close()


print(
    "PR curve saved to "
    "reports/pr_curve.png"
)


# =========================================================
# 14. SAVE INFERENCE CONFIGURATION
# =========================================================

Path("models").mkdir(
    exist_ok=True
)

config = {
    "model": "resnet18_day12_best.pth",
    "task": "healthy_vs_diseased",
    "healthy_class": "healthy",
    "disease_classes": [
        "early_blight",
        "late_blight",
        "leaf_mold"
    ],
    "threshold": chosen_threshold,
    "precision": chosen_precision,
    "recall": chosen_recall,
    "target_recall": 0.95,
    "target_precision": 0.80
}


with open(
    "models/inference_config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


print(
    "Inference configuration saved to "
    "models/inference_config.json"
)


print("\nDay 13 evaluation complete!")