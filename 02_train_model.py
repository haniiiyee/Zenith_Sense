import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
PATHS = ['./dataset']

def train_model():
    data_dir = None
    for p in PATHS:
        if os.path.exists(p):
            data_dir = p
            break
    
    if not data_dir:
        print("❌ ERROR: Could not find dataset folder.")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ])

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # FIX: Unfreeze the base model so it can learn the difference between fire and leaves
    base_model.trainable = True 

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # FIX: Extremely low learning rate for safe fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    print("🚀 Starting Fine-Tuning Pipeline...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/spacenet.keras')
    print("✅ Fine-Tuned MobileNetV3 SpaceNet Model Saved successfully.")

if __name__ == "__main__":
    train_model()