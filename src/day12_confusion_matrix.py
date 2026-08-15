from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# 1. LOAD SAVED PREDICTIONS
# =========================================================

data = torch.load(
    "reports/day12_predictions.pt",
    weights_only=False
)

y_true = np.array(data["y_true"])
y_pred = np.array(data["y_pred"])
class_names = data["class_names"]


# =========================================================
# 2. CREATE CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("Confusion Matrix:")
print(cm)


# =========================================================
# 3. CALCULATE PER-CLASS ACCURACY
# =========================================================

print("\nPer-class accuracy:")

per_class_accuracy = []

for i, class_name in enumerate(class_names):

    total = cm[i].sum()

    correct = cm[i, i]

    accuracy = correct / total

    per_class_accuracy.append(
        accuracy
    )

    print(
        f"{class_name}: "
        f"{accuracy * 100:.2f}%"
    )


# =========================================================
# 4. OVERALL ACCURACY
# =========================================================

overall_accuracy = (
    np.trace(cm) / np.sum(cm)
)

print(
    f"\nOverall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)


# =========================================================
# 5. CREATE REPORTS FOLDER
# =========================================================

Path("reports").mkdir(
    exist_ok=True
)


# =========================================================
# 6. PLOT CONFUSION MATRIX
# =========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "ResNet18 Confusion Matrix - Day 12 Test Set"
)

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix_day12.png",
    dpi=300
)

plt.close()


# =========================================================
# 7. SAVE PER-CLASS METRICS
# =========================================================

with open(
    "reports/metrics_day12.md",
    "w"
) as f:

    f.write("# Day 12 ResNet18 Classification Metrics\n\n")

    f.write("## Overall Accuracy\n\n")

    f.write(
        f"**Test Accuracy: "
        f"{overall_accuracy * 100:.2f}%**\n\n"
    )

    f.write("## Per-Class Accuracy\n\n")

    f.write(
        "| Class | Correct | Total | Accuracy |\n"
    )

    f.write(
        "|---|---:|---:|---:|\n"
    )

    for i, class_name in enumerate(class_names):

        total = cm[i].sum()

        correct = cm[i, i]

        accuracy = (
            correct / total * 100
        )

        f.write(
            f"| {class_name} | "
            f"{correct} | "
            f"{total} | "
            f"{accuracy:.2f}% |\n"
        )

print(
    "\nConfusion matrix saved to "
    "reports/confusion_matrix_day12.png"
)

print(
    "Metrics saved to "
    "reports/metrics_day12.md"
)