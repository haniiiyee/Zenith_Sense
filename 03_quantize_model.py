import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/spacenet.keras'
DATA_DIR = './dataset'
IMG_SIZE = 224
OUTPUT_PATH = 'models/zenith_int8.tflite'

def quantize_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find {MODEL_PATH}. Did training finish?")
        return

    print(f"⏳ Loading SpaceNet model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 1. Initialize Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Build Representative Dataset (Critical for INT8 calibration)
    print("🔬 Building representative dataset for 8-bit calibration...")
    def representative_data_gen():
        dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=10,
            shuffle=True
        )
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(lambda x, y: normalization_layer(x))
        
        for input_value in dataset.take(10): # Calibrate using 100 images
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    
    # 3. Force INT8 parameters
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32  
    converter.inference_output_type = tf.float32 

    # 4. Convert and Save
    print("🗜️ Compressing model to INT8 (This may take a minute)...")
    tflite_quant_model = converter.convert()
    
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_quant_model)
        
    # 5. Verify Constraints
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print("\n" + "="*40)
    print("✅ QUANTIZATION COMPLETE")
    print(f"💾 Final Model Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 2.0:
        print("🚀 FLIGHT READY: Meets CubeSat SWaP constraints!")
    else:
        print("❌ WARNING: Model exceeds hardware memory limits.")
    print("="*40)

if __name__ == "__main__":
    quantize_model()