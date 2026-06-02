import tensorflow as tf
import numpy as np
import os

def representative_data_gen():
    paths = ['./dataset/fire_dataset', './dataset/the_wildfire_dataset/the_wildfire_dataset']
    dataset_path = next((p for p in paths if os.path.exists(p)), None)
    
    if not dataset_path: return

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, image_size=(224, 224), batch_size=1
    )
    # Take 50 images, normalize to 0-1
    for image, _ in ds.take(50):
        yield [tf.cast(image / 255.0, tf.float32)]

def quantize():
    model_path = 'models/spacenet.keras'  # POINTING TO NEW MODEL
    output_path = 'models/zenith_int8.tflite'

    if not os.path.exists(model_path):
        print("Error: spacenet.keras not found. Run script 02!")
        return

    print("Loading SpaceNet...")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    print("Compressing...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    quantize()