from PIL import Image
import os

def convert_and_rename(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    files = sorted([f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in image_extensions])

    for idx, fname in enumerate(files, start=1):
        old_path = os.path.join(folder_path, fname)
        new_name = f"{idx:03}.jpg"
        new_path = os.path.join(folder_path, new_name)

        try:
            with Image.open(old_path) as img:
                rgb_img = img.convert("RGB")  # Convert to RGB if needed
                rgb_img.save(new_path, "JPEG")
        except Exception as e:
            print(f" Error converting {old_path}: {e}")
            continue

        if old_path != new_path:
            os.remove(old_path)  # Remove old file if name/format changed

        print(f" Saved: {new_name}")

# Paths
base_dir = "training/data"
folders = ["apple", "banana", "other"]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        print(f"\n Processing {folder_path}")
        convert_and_rename(folder_path)
    else:
        print(f" Skipped missing folder: {folder_path}")