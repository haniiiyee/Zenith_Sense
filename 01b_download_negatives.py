import os
import shutil
from bing_image_downloader import downloader

# --- CONFIGURATION ---
TARGET_FOLDER = "./dataset/safe"
TEMP_FOLDER = "./dataset/temp_downloads"

# These specific search terms force the AI to look at orange/red geometries that are NOT fires
QUERIES = [
    "autumn forest orange leaves landscape",
    "bright red sunset over trees",
    "orange desert sand dunes"
]
IMAGES_PER_QUERY = 75  # 3 queries * 75 = 225 new hard negatives

def inject_hard_negatives():
    print("Starting automated injection of Hard Negatives...")
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"Error: {TARGET_FOLDER} does not exist. Run your main dataset setup first.")
        return

    # 1. Download images into a temporary staging area
    for query in QUERIES:
        print(f"\n🔍 Scraping: '{query}'")
        try:
            downloader.download(
                query, 
                limit=IMAGES_PER_QUERY,  
                output_dir=TEMP_FOLDER, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=60,
                verbose=False
            )
        except Exception as e:
            print(f"Warning: Could not complete download for {query}. Error: {e}")

    # 2. Flatten the folder structure and move to the 'safe' training directory
    print("\nIntegrating images into the /safe/ training directory...")
    moved_count = 0
    
    for root, dirs, files in os.walk(TEMP_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                
                # Prefix the filename to avoid accidentally overwriting existing dataset files
                new_filename = f"hard_neg_{moved_count}_{file}"
                dest_path = os.path.join(TARGET_FOLDER, new_filename)
                
                shutil.move(src_path, dest_path)
                moved_count += 1

    # 3. Clean up the staging area
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)

    print(f"\nSUCCESS: {moved_count} hard negative images injected into {TARGET_FOLDER}.")
    print("The dataset is now ready for retraining.")

if __name__ == "__main__":
    inject_hard_negatives()