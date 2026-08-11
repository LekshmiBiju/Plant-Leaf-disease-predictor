from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import LeafDiseaseDataset
from transforms import val_transform
from model import LeafDiseaseCNN


# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. VALIDATION DATASET
# --------------------------------------------------

val_ds = LeafDiseaseDataset(
    "data/val",
    transform=val_transform
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(
    "Validation images:",
    len(val_ds)
)


# --------------------------------------------------
# 3. LOAD BEST MODEL
# --------------------------------------------------

model = LeafDiseaseCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/leaf_cnn_best.pth",
        map_location=device
    )
)

model.eval()

print(
    "Best model loaded successfully."
)


# --------------------------------------------------
# 4. COLLECT PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


# --------------------------------------------------
# 5. PER-CLASS RECALL
# --------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    output_dict=True,
    zero_division=0
)


print("\nPer-class recall:")
print("=================")

recall_values = {}

for class_id in sorted(
    set(all_labels)
):

    class_name = str(class_id)

    recall = report[
        class_name
    ]["recall"]

    recall_values[class_name] = recall

    print(
        f"Class {class_name}: "
        f"{recall:.4f}"
    )


# --------------------------------------------------
# 6. SAVE RECALL REPORT
# --------------------------------------------------

Path("reports").mkdir(
    exist_ok=True
)

with open(
    "reports/per_class_recall.txt",
    "w"
) as f:

    f.write(
        "Per-Class Recall - Day 8\n"
    )

    f.write(
        "========================\n\n"
    )

    for class_name, recall in recall_values.items():

        f.write(
            f"Class {class_name}: "
            f"{recall:.4f}\n"
        )

    f.write(
        "\nOverall metrics:\n"
    )

    f.write(
        f"Accuracy: "
        f"{report['accuracy']:.4f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{report['macro avg']['recall']:.4f}\n"
    )


print(
    "\nRecall report saved to "
    "reports/per_class_recall.txt"
)