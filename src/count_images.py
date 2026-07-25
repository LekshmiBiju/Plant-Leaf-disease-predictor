from pathlib import Path

train_path = Path("data/train")

print("Images per class:\n")

for class_folder in train_path.iterdir():
    if class_folder.is_dir():
        count = len(list(class_folder.glob("*.jpg")))
        print(f"{class_folder.name}: {count}")