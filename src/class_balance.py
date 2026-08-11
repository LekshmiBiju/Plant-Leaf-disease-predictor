from pathlib import Path
import csv

TRAIN_DIR = Path("data/train")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

rows = []

for class_dir in sorted(TRAIN_DIR.iterdir()):
    if class_dir.is_dir():
        count = len([
            p for p in class_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        rows.append((class_dir.name, count))

with open(REPORT_DIR / "class_balance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "count"])
    writer.writerows(rows)

print("Class distribution:")
for class_name, count in rows:
    print(f"{class_name}: {count}")

print("\nSaved to reports/class_balance.csv")