from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
from torchvision.models import resnet18

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

print(
    "Test images:",
    len(test_ds)
)


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
# 5. FIND MISCLASSIFIED IMAGES
# =========================================================

misclassified = []


with torch.no_grad():

    for idx in range(len(test_ds)):

        image_tensor, true_label = test_ds[idx]

        image_input = image_tensor.unsqueeze(
            0
        ).to(device)

        output = model(
            image_input
        )

        predicted_label = output.argmax(
            dim=1
        ).item()

        if predicted_label != true_label:

            image_path = test_ds.samples[idx][0]

            misclassified.append(
                (
                    image_path,
                    true_label,
                    predicted_label
                )
            )


print(
    "Misclassified images:",
    len(misclassified)
)


# =========================================================
# 6. CREATE REPORTS FOLDER
# =========================================================

Path("reports").mkdir(
    exist_ok=True
)


# =========================================================
# 7. CREATE GALLERY
# =========================================================

if len(misclassified) == 0:

    print(
        "No misclassified images found."
    )

else:

    number_of_images = len(
        misclassified
    )

    fig, axes = plt.subplots(
        1,
        number_of_images,
        figsize=(
            6 * number_of_images,
            6
        )
    )

    # Handle case where there is only 1 image
    if number_of_images == 1:
        axes = [axes]

    for ax, item in zip(
        axes,
        misclassified
    ):

        image_path = item[0]
        true_label = item[1]
        predicted_label = item[2]

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)

        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Predicted: {CLASS_NAMES[predicted_label]}",
            fontsize=12
        )

        ax.axis("off")


    plt.tight_layout()

    plt.savefig(
        "reports/day12_misclassification_gallery.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Gallery saved to "
        "reports/day12_misclassification_gallery.png"
    )


print("\nGallery generation complete!")