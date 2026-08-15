import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from dataset import LeafDiseaseDataset
from transforms import val_transform


# =========================================================
# SETTINGS
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]


# =========================================================
# DATASET
# =========================================================

test_ds = LeafDiseaseDataset(
    "data/day12/test",
    transform=val_transform
)


# =========================================================
# MODEL
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
# ERROR FOLDER
# =========================================================

error_dir = Path(
    "reports/errors"
)

error_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FIND ERRORS
# =========================================================

errors = []


with torch.no_grad():

    for idx in range(len(test_ds)):

        image_tensor, true_label = test_ds[idx]

        output = model(
            image_tensor.unsqueeze(0).to(device)
        )

        predicted_label = output.argmax(
            dim=1
        ).item()

        if predicted_label != true_label:

            image_path = test_ds.samples[idx][0]

            errors.append(
                (
                    image_path,
                    true_label,
                    predicted_label
                )
            )


# =========================================================
# COPY ERRORS
# =========================================================

for number, (
    image_path,
    true_label,
    predicted_label
) in enumerate(errors, start=1):

    new_name = (
        f"{number}_"
        f"true_{CLASS_NAMES[true_label]}_"
        f"pred_{CLASS_NAMES[predicted_label]}_"
        f"{image_path.name}"
    )

    shutil.copy2(
        image_path,
        error_dir / new_name
    )

    print(
        f"Copied: {new_name}"
    )


print(
    f"\nTotal misclassified images: "
    f"{len(errors)}"
)
