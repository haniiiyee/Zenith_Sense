import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2

# --- CONFIGURATION ---
IMG_SIZE = 224 
MODEL_PATH = 'models/zenith_int8.tflite'
DATASET_ROOT = './dataset'

def predict_random():
    # 1. SETUP PATHS FOR NEW BALANCED DATASET
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root folder '{DATASET_ROOT}' not found.")
        return

    # Randomly select between testing a 'fire' or 'safe' image
    target_class = random.choice(['fire', 'safe'])
    class_dir = os.path.join(DATASET_ROOT, target_class)
    
    if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
        print(f"Error: No images found in {class_dir}")
        return

    random_file = random.choice(os.listdir(class_dir))
    img_path = os.path.join(class_dir, random_file)
    print(f"Testing Random File From [{target_class.upper()}]: {random_file}")

    # 2. LOAD TFLITE INTERPRETER
    try:
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    except Exception:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. PROCESSING IMAGES
    # High-Quality Display Image
    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # Edge-AI Pipeline Preprocessing
    img_ai = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    input_data = (np.expand_dims(img_array, axis=0) / 255.0).astype(np.float32)
    
    # 4. RUN INFERENCE
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_prediction = output_data[0][0]

    # 5. MAPPING LOGIC (0 = FIRE, 1 = SAFE)
    if raw_prediction < 0.5:
        label = "FIRE DETECTED"
        color = "red"
        confidence = (1.0 - raw_prediction) * 100
    else:
        label = "SAFE"
        color = "green"
        confidence = raw_prediction * 100

    print(f"(Result): {label} ({confidence:.2f}%)")
    
    # 6. VISUALIZATION
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f"Target Class: {target_class.upper()} | Prediction: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_random()