import os
import shutil
import random
import kagglehub

# 1. Download dataset using kagglehub
print("Downloading dataset...")
path = kagglehub.dataset_download("feyzazkefe/trashnet")
print("Dataset downloaded to:", path)

# 2. Define paths
# Note: Check the exact folder name inside 'path' if it has a subfolder like 'dataset-resized'
source_dir = path  # Update if the images are inside a subfolder (e.g., os.path.join(path, 'dataset-resized'))

train_dir = "dataset/train"
val_dir = "dataset/val"

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Create train/val directories
for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

# 3. Split files (80% train, 20% val)
split_ratio = 0.8

for c in classes:
    # Look for class folder in download path (handling potential nested folders)
    class_path = None
    for root, dirs, files in os.walk(source_dir):
        if c in dirs:
            class_path = os.path.join(root, c)
            break
            
    if not class_path:
        print(f"Could not find folder for class: {c}")
        continue
        
    images = os.listdir(class_path)
    random.shuffle(images)
    
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # Copy to train
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, c, img)
        if os.path.isfile(src):
            shutil.copy(src, dst)
            
    # Copy to val
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, c, img)
        if os.path.isfile(src):
            shutil.copy(src, dst)

print("Dataset successfully set up and split into dataset/train and dataset/val!")