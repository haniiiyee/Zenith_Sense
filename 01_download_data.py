import os
import zipfile
import shutil

# --- CONFIGURATION (BALANCED DATASET) ---
# Swapped to a mathematically balanced 50/50 dataset
DATASET_SLUG = "alik05/forest-fire-dataset"
DOWNLOAD_PATH = "./dataset"
ZIP_FILE = f"{DOWNLOAD_PATH}/forest-fire-dataset.zip"

def setup_dataset():
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    final_fire_dir = os.path.join(DOWNLOAD_PATH, 'fire')
    final_safe_dir = os.path.join(DOWNLOAD_PATH, 'safe')

    # Skip if already formatted
    if os.path.exists(final_fire_dir) and os.path.exists(final_safe_dir):
        print("Balanced dataset already exists and is formatted.")
        return

    print(f"Downloading Balanced Dataset ({DATASET_SLUG})...")
    exit_code = os.system(f"kaggle datasets download -d {DATASET_SLUG} -p {DOWNLOAD_PATH}")

    if exit_code != 0:
        print(f"Error: Kaggle download failed. Please check your Kaggle configuration.")
        return

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_PATH)

    os.remove(ZIP_FILE)
    
    print("Reorganizing folders for the training pipeline...")
    os.makedirs(final_fire_dir, exist_ok=True)
    os.makedirs(final_safe_dir, exist_ok=True)

    # Route images to clean /fire/ and /safe/ directories based on original folder names
    for root, dirs, files in os.walk(DOWNLOAD_PATH):
        
        # Normalize paths to prevent re-processing the destination folders
        norm_root = os.path.normpath(root)
        if norm_root == os.path.normpath(final_fire_dir) or norm_root == os.path.normpath(final_safe_dir):
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                
                # Check the original folder name to determine the class
                if 'non_fire' in root.lower() or 'safe' in root.lower():
                    shutil.move(src_path, os.path.join(final_safe_dir, file))
                elif 'fire' in root.lower():
                    shutil.move(src_path, os.path.join(final_fire_dir, file))

    print(f"Dataset setup complete. Data is balanced and stored in {DOWNLOAD_PATH}/fire and {DOWNLOAD_PATH}/safe.")

if __name__ == "__main__":
    setup_dataset()