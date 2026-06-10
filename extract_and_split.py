import os
import zipfile
import random
import shutil

# Path configurations
zip_path = r"C:\Users\emink\Downloads\archive.zip"
project_dir = r"c:\Users\emink\Desktop\projem\analysis-main (1)\analysis-main"
dataset_dir = os.path.join(project_dir, "dataset")

train_split_dir = os.path.join(dataset_dir, "train_split")
val_split_dir = os.path.join(dataset_dir, "val_split")

# Output subdirectories
train_normal_dir = os.path.join(train_split_dir, "normal")
train_pneu_dir = os.path.join(train_split_dir, "viral_pneumonia")
val_normal_dir = os.path.join(val_split_dir, "normal")
val_pneu_dir = os.path.join(val_split_dir, "viral_pneumonia")

def clean_dir(directory):
    if os.path.exists(directory):
        print(f"Cleaning existing files in {directory}...")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory, exist_ok=True)

def make_unique_name(zip_filepath):
    # E.g., 'chest_xray/train/NORMAL/IM-0001-0001.jpeg' -> 'train_NORMAL_IM-0001-0001.jpeg'
    parts = zip_filepath.split('/')
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
    return parts[-1]

def main():
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found!")
        return

    print("Reading zip archive...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        
        # Filter files to ignore macOS system files and non-images
        all_files = [
            f for f in namelist 
            if "__MACOSX" not in f 
            and "._" not in f.split("/")[-1] 
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        normal_images = [f for f in all_files if "NORMAL" in f]
        # Only take viral pneumonia (images with 'virus' in their filename)
        viral_pneu_images = [f for f in all_files if "PNEUMONIA" in f and "virus" in f.lower()]
        
        print(f"Found {len(normal_images)} Normal images in zip.")
        print(f"Found {len(viral_pneu_images)} Viral Pneumonia images in zip.")
        
        # Shuffle for a random split
        random.seed(42)
        random.shuffle(normal_images)
        random.shuffle(viral_pneu_images)
        
        # Split ratios: 80% train, 20% validation
        normal_split_idx = int(len(normal_images) * 0.8)
        normal_train = normal_images[:normal_split_idx]
        normal_val = normal_images[normal_split_idx:]
        
        pneu_split_idx = int(len(viral_pneu_images) * 0.8)
        pneu_train = viral_pneu_images[:pneu_split_idx]
        pneu_val = viral_pneu_images[pneu_split_idx:]
        
        # Clean target directories (keeping 'covid' untouched)
        clean_dir(train_normal_dir)
        clean_dir(train_pneu_dir)
        clean_dir(val_normal_dir)
        clean_dir(val_pneu_dir)
        
        # Extract files using unique names to prevent overwriting
        print("\nExtracting Normal train images...")
        for f in normal_train:
            unique_name = make_unique_name(f)
            target = os.path.join(train_normal_dir, unique_name)
            with open(target, 'wb') as out_f:
                out_f.write(z.read(f))
                
        print("Extracting Normal validation images...")
        for f in normal_val:
            unique_name = make_unique_name(f)
            target = os.path.join(val_normal_dir, unique_name)
            with open(target, 'wb') as out_f:
                out_f.write(z.read(f))
                
        print("Extracting Viral Pneumonia train images...")
        for f in pneu_train:
            unique_name = make_unique_name(f)
            target = os.path.join(train_pneu_dir, unique_name)
            with open(target, 'wb') as out_f:
                out_f.write(z.read(f))
                
        print("Extracting Viral Pneumonia validation images...")
        for f in pneu_val:
            unique_name = make_unique_name(f)
            target = os.path.join(val_pneu_dir, unique_name)
            with open(target, 'wb') as out_f:
                out_f.write(z.read(f))
                
    print("\nExtraction and split complete!")
    print(f"Train - Normal: {len(os.listdir(train_normal_dir))} images")
    print(f"Train - Viral Pneumonia: {len(os.listdir(train_pneu_dir))} images")
    print(f"Val - Normal: {len(os.listdir(val_normal_dir))} images")
    print(f"Val - Viral Pneumonia: {len(os.listdir(val_pneu_dir))} images")

if __name__ == "__main__":
    main()
