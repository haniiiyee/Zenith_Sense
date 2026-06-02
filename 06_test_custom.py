import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
IMG_SIZE = 224
MODEL_PATH = 'models/zenith_int8.tflite'
# Change this path to point exactly to your downloaded test image file
CUSTOM_IMAGE_PATH = 'download.jpeg' 

def test_custom_image():
    if not os.path.exists(CUSTOM_IMAGE_PATH):
        print(f"❌ Error: Custom image not found at '{CUSTOM_IMAGE_PATH}'. Check file name or extension.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Quantized model not found at '{MODEL_PATH}'. Run script 03 first.")
        return

    print(f"🖼️ Evaluating Custom Target: {CUSTOM_IMAGE_PATH}")

    # 1. Load Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Image Pipelines
    img_display = cv2.imread(CUSTOM_IMAGE_PATH)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    img_ai = tf.keras.utils.load_img(CUSTOM_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    input_data = (np.expand_dims(img_array, axis=0) / 255.0).astype(np.float32)

    # 3. Inference Execution
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 4. Calibration Analysis
    if raw_prediction < 0.5:
        label = "FIRE DETECTED"
        color = "red"
        confidence = (1.0 - raw_prediction) * 100
    else:
        label = "SAFE"
        color = "green"
        confidence = raw_prediction * 100

    print(f"Analysis Verdict: {label} [{confidence:.2f}% Confidence]")

    # 5. Render Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f"Custom Test File Evaluation\nVerdict: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_custom_image()