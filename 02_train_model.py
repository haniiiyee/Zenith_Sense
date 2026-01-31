import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
IMG_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 10
# We check for both possible folder names
PATHS = [
    './dataset/fire_dataset',
    './dataset/the_wildfire_dataset/the_wildfire_dataset'
]

def train_model():
    # 1. Find the correct dataset path
    data_dir = None
    for p in PATHS:
        if os.path.exists(p):
            data_dir = p
            break
    
    if not data_dir:
        print("❌ ERROR: Could not find dataset folder.")
        return

    print(f"✅ Loading Data from: {data_dir}")

    # 2. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    # 3. CHECK CLASS NAMES (Important!)
    class_names = train_ds.class_names
    print(f"🧐 CLASSES FOUND: {class_names}")
    # Usually: ['fire_images', 'non_fire_images']
    # This means 0 = FIRE, 1 = SAFE. 
    # We will need to remember this for the predictor!

    # 4. Standardize (0-1)
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 5. Build SpaceNet
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("🚀 Starting Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/spacenet.keras')
    print("✅ SpaceNet Model Saved successfully.")

if __name__ == "__main__":
    train_model()