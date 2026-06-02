import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
IMG_SIZE = 224
KERAS_MODEL_PATH = 'models/spacenet.keras'
CUSTOM_IMAGE_PATH = 'download.jpeg' # Ensure this is your autumn lake image

def test_raw_keras():
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ Error: Raw model not found at '{KERAS_MODEL_PATH}'.")
        return

    print(f"🧠 Loading Uncompressed FP32 Keras Model...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    # 1. Image Pipelines
    img_display = cv2.imread(CUSTOM_IMAGE_PATH)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    img_ai = tf.keras.utils.load_img(CUSTOM_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    input_data = (np.expand_dims(img_array, axis=0) / 255.0).astype(np.float32)

    # 2. Inference Execution
    print("⚙️ Running Inference...")
    raw_prediction = model.predict(input_data, verbose=0)[0][0]

    # 3. Calibration Analysis
    if raw_prediction < 0.5:
        label = "FIRE DETECTED"
        color = "red"
        confidence = (1.0 - raw_prediction) * 100
    else:
        label = "SAFE"
        color = "green"
        confidence = raw_prediction * 100

    print(f"\n📊 RAW FP32 RESULT: {label} [{confidence:.2f}% Confidence]")

    # 4. Render Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f"Uncompressed FP32 Model Evaluation\nVerdict: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_raw_keras()