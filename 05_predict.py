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
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Error: Dataset root folder '{DATASET_ROOT}' not found.")
        return

    target_class = random.choice(['fire', 'safe'])
    class_dir = os.path.join(DATASET_ROOT, target_class)
    
    random_file = random.choice(os.listdir(class_dir))
    img_path = os.path.join(class_dir, random_file)

    # Disable XNNPACK delegate to prevent HardSwish INT8 crash
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    img_ai = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    
    # FIX: Natively ingest [0, 255] tensor (NO DIVISION)
    input_data = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if raw_prediction < 0.5:
        label = "FIRE DETECTED"
        color = "red"
        confidence = (1.0 - raw_prediction) * 100
    else:
        label = "SAFE"
        color = "green"
        confidence = raw_prediction * 100

    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f"Target Class: {target_class.upper()} | Prediction: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_random()