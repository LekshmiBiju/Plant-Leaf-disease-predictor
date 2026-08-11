import os
import matplotlib.pyplot as plt
from PIL import Image

from transforms import train_transform


# Pick one image from the training dataset
image_path = None

for class_name in ["early_blight", "healthy", "late_blight", "leaf_mold"]:
    class_dir = os.path.join("data", "train", class_name)

    if os.path.exists(class_dir):
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if files:
            image_path = os.path.join(class_dir, files[0])
            break


if image_path is None:
    raise FileNotFoundError("No image found in data/train")


# Load original image
image = Image.open(image_path).convert("RGB")


# Generate 8 augmented versions
augmented_images = []

for _ in range(8):
    augmented = train_transform(image)

    # Undo normalization for visualization
    mean = augmented.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = augmented.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    augmented = augmented * std + mean
    augmented = augmented.clamp(0, 1)

    augmented_images.append(augmented.permute(1, 2, 0).numpy())


# Create 2 x 4 grid
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, img in zip(axes.flat, augmented_images):
    ax.imshow(img)
    ax.axis("off")

plt.suptitle("Data Augmentation Samples")

os.makedirs("reports", exist_ok=True)

plt.tight_layout()
plt.savefig(
    "reports/augmentation_samples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Augmentation samples saved to reports/augmentation_samples.png")
