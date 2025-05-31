import os
import shutil
from sklearn.model_selection import train_test_split

def split_images(class_dir, output_dir, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert train_size + val_size + test_size == 1.0, "Splits must sum to 1.0"
    all_images = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    train_val, test = train_test_split(all_images, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size/(train_size + val_size), random_state=seed)

    for split_name, split_list in zip(["train", "val", "test"], [train, val, test]):
        split_class_dir = os.path.join(output_dir, split_name, os.path.basename(class_dir))
        os.makedirs(split_class_dir, exist_ok=True)
        for fname in split_list:
            shutil.copy(os.path.join(class_dir, fname), os.path.join(split_class_dir, fname))
        print(f"Copied {len(split_list)} files to {split_class_dir}")

if __name__ == "__main__":
    base_input = "training/data"
    base_output = "training/dataset"

    for class_name in ["banana", "apple"]:
        class_path = os.path.join(base_input, class_name)
        split_images(class_path, base_output)