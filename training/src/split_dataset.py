import os
import random
import shutil
from pathlib import Path

random.seed(42)  # for reproducibility

# Source and target root directories
source_root = Path("training/data")
target_root = Path("training/dataset")

# Desired split percentages
splits = {"train": 0.7, "val": 0.15, "test": 0.15}

# Clean target root
if target_root.exists():
    shutil.rmtree(target_root)
target_root.mkdir(parents=True)

# Go through each class folder (banana, apple, other, etc.)
for class_folder in source_root.iterdir():
    if not class_folder.is_dir():
        continue
    class_name = class_folder.name
    images = list(class_folder.glob("*.jpg"))

    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * splits["train"])
    n_val = int(n_total * splits["val"])
    n_test = n_total - n_train - n_val

    split_counts = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in split_counts.items():
        dest_dir = target_root / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, dest_dir)
        print(f"Copied {len(files)} files to {dest_dir}")