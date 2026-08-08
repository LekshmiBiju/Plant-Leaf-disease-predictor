from pathlib import Path
import random
import shutil

random.seed(42)

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

CLASSES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold"
]

VAL_RATIO = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for class_name in CLASSES:
    train_class = TRAIN_DIR / class_name
    val_class = VAL_DIR / class_name

    val_class.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in train_class.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)

    val_images = images[:val_count]

    print(f"{class_name}: {len(images)} total")
    print(f"Moving {val_count} images to validation set")

    for image in val_images:
        shutil.move(str(image), str(val_class / image.name))

print("\nDataset split completed!")