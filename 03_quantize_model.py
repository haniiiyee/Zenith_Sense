import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/spacenet.keras'
DATA_DIR = './dataset'
IMG_SIZE = 224
OUTPUT_PATH = 'models/zenith_int8.tflite'

def quantize_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=10, shuffle=True
        )
        for input_value, _ in dataset.take(10):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32  
    converter.inference_output_type = tf.float32 

    print("🗜️ Compressing model to INT8...")
    tflite_quant_model = converter.convert()
    
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_quant_model)
        
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✅ QUANTIZATION COMPLETE. Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    quantize_model()