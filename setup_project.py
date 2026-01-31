import os

# --- PROJECT CONFIGURATION ---
project_name = "Zenith_Sense"

# --- FILE CONTENTS (Updated for Lite Dataset) ---
requirements_txt = """tensorflow>=2.15.0
numpy
matplotlib
kaggle
psutil
"""

script_01_download = """import os
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
"""

script_02_train = """import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
# Note: This dataset has folders 'fire_images' and 'non_fire_images'
DATA_DIR = './dataset/fire_dataset'

def train_model():
    if not os.path.exists(DATA_DIR):
        print("Error: Dataset not found. Run script 01 first.")
        return

    print("Loading Lite Dataset...")
    # Smart Loading
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    # Normalization
    norm_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (norm_layer(x), y))

    # Architecture
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/mobilenet_v3.keras')
    print("Model saved to models/mobilenet_v3.keras")

if __name__ == "__main__":
    train_model()
"""

script_03_quantize = """import tensorflow as tf
import numpy as np
import os

def representative_data_gen():
    dataset_path = './dataset/fire_dataset'
    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, image_size=(224, 224), batch_size=1
    )
    for image, _ in ds.take(50):
        yield [tf.cast(image / 255.0, tf.float32)]

def quantize():
    model_path = 'models/mobilenet_v3.keras'
    output_path = 'models/zenith_int8.tflite'

    if not os.path.exists(model_path):
        print("Error: Train the model first.")
        return

    print("Loading Float32 Model...")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("Quantizing...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    quantize()
"""

script_04_benchmark = """import tensorflow as tf
import numpy as np
import time
import tracemalloc
import os

def benchmark_tflite(model_path):
    print(f"--- Benchmarking: {model_path} ---")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    input_shape = input_details[0]['shape']
    dummy_input = np.random.randint(0, 255, input_shape).astype(np.uint8)

    tracemalloc.start()
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    start_time = time.time()
    for _ in range(50):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    end_time = time.time()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_time = ((end_time - start_time) / 50) * 1000
    peak_ram_mb = peak / (1024 * 1024)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Peak RAM : {peak_ram_mb:.2f} MB")
    print(f"Avg Time : {avg_time:.2f} ms")
    
    if peak_ram_mb < 50:
        print("Verdict: [PASS] Fits in 50MB Space-Grade Memory.")
    else:
        print("Verdict: [FAIL] Memory usage too high.")

if __name__ == "__main__":
    if os.path.exists('models/zenith_int8.tflite'):
        benchmark_tflite('models/zenith_int8.tflite')
    else:
        print("Model not found. Run script 03 first.")
"""

# --- MAIN GENERATOR LOGIC ---
def create_project():
    # Write Files
    files = {
        "requirements.txt": requirements_txt,
        "01_download_data.py": script_01_download,
        "02_train_model.py": script_02_train,
        "03_quantize_model.py": script_03_quantize,
        "04_benchmark.py": script_04_benchmark
    }
    
    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated file: {filename}")

    print("\n--- LITE MODE ACTIVATED ---")
    print("Run 'python 01_download_data.py' to download the smaller dataset.")

if __name__ == "__main__":
    create_project()