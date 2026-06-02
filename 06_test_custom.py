import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
IMG_SIZE = 224
MODEL_PATH = 'models/zenith_int8.tflite'
CUSTOM_IMAGE_PATH = './download.jpeg' 

def test_custom_image():
    if not os.path.exists(CUSTOM_IMAGE_PATH):
        print("❌ Error: Custom image not found.")
        return

    # Disable XNNPACK delegate to prevent INT8 HardSwish crash
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_display = cv2.imread(CUSTOM_IMAGE_PATH)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    img_ai = tf.keras.utils.load_img(CUSTOM_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_ai)
    
    # Ingesting raw [0, 255] tensor
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

    print(f"Verdict: {label} [{confidence:.2f}% Confidence]")

    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f"Verdict: {label} ({confidence:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_custom_image()