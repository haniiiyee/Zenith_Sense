import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2 # We need OpenCV for better image loading

# --- CONFIGURATION ---
IMG_SIZE = 224 # What the AI sees

def predict_fire():
    # 1. SETUP PATHS
    base_dir = './dataset/fire_dataset/fire_images'
    if not os.path.exists(base_dir): 
        base_dir = './dataset/the_wildfire_dataset/the_wildfire_dataset/test/fire'
    
    if not os.path.exists(base_dir):
        print("Error: No images found.")
        return

    random_file = random.choice(os.listdir(base_dir))
    img_path = os.path.join(base_dir, random_file)
    print(f"Testing: {random_file}")

    # 2. LOAD MODEL
    try:
        interpreter = tf.lite.Interpreter(
            model_path='models/zenith_int8.tflite',
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    except:
        interpreter = tf.lite.Interpreter(model_path='models/zenith_int8.tflite')

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- THE COSMETIC FIX ---
    
    # A. Load High-Quality Image for DISPLAY
    # OpenCV loads as BGR, convert to RGB for matplotlib
    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # B. Load Low-Quality Image for AI
    img_ai = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    input_data = (np.expand_dims(img_array, axis=0) / 255.0).astype(np.float32)
    
    # 4. RUN AI
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_prediction = output_data[0][0]

    # 5. RESULT LOGIC
    if raw_prediction < 0.5:
        label = "FIRE DETECTED"
        color = "red"
        confidence = (1.0 - raw_prediction) * 100
    else:
        label = "SAFE"
        color = "green"
        confidence = raw_prediction * 100

    print(f"Result: {label} ({confidence:.2f}%)")
    
    # SHOW THE HIGH QUALITY IMAGE
    plt.figure(figsize=(8, 8)) # Make the window bigger
    plt.imshow(img_display)
    plt.title(f"AI Input: {IMG_SIZE}x{IMG_SIZE} | Display: Original HQ\nPrediction: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_fire()