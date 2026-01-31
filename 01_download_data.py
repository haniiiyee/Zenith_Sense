import os
import zipfile

# --- CONFIGURATION (LITE VERSION) ---
# Using 'phylake1337/fire-dataset' (380MB) instead of the 10GB one
DATASET_SLUG = "phylake1337/fire-dataset"
DOWNLOAD_PATH = "./dataset"
ZIP_FILE = f"{DOWNLOAD_PATH}/fire-dataset.zip"

def setup_dataset():
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    # Check if data exists
    if not os.path.exists(os.path.join(DOWNLOAD_PATH, 'fire_dataset')):
        print(f"Downloading Lite Dataset ({DATASET_SLUG})...")
        
        # Download
        exit_code = os.system(f"kaggle datasets download -d {DATASET_SLUG} -p {DOWNLOAD_PATH}")
        
        if exit_code != 0:
            print("Error: Kaggle download failed. Please download 'phylake1337/fire-dataset' manually.")
            return

        print("Unzipping...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_PATH)
        
        # Cleanup
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
            
        print("Dataset setup complete.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    setup_dataset()
