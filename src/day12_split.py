import os
import random
import shutil

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/day12"

CLASSES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)


# Create output directories
for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )


for class_name in CLASSES:

    source_dir = os.path.join(RAW_DIR, class_name)

    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    # Copy training images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_dir, image),
            os.path.join(OUTPUT_DIR, "train", class_name, image)
        )

    # Copy validation images
    for image in val_images:
        shutil.copy2(
            os.path.join(source_dir, image),
            os.path.join(OUTPUT_DIR, "val", class_name, image)
        )

    # Copy test images
    for image in test_images:
        shutil.copy2(
            os.path.join(source_dir, image),
            os.path.join(OUTPUT_DIR, "test", class_name, image)
        )

print("\nDay 12 dataset split completed successfully.")