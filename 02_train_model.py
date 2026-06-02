import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
IMG_SIZE = 224  # Maintained at 224 to ensure memory footprint fits within <2MB RAM limits
BATCH_SIZE = 32
EPOCHS = 10
PATHS = [
    './dataset'  # Fixed syntax error: cleared invalid assignment within list literal
]

def train_model():
    # 1. Find the correct dataset path
    data_dir = None
    for p in PATHS:
        if os.path.exists(p):
            data_dir = p
            break
    
    if not data_dir:
        print("ERROR: Could not find dataset folder.")
        return

    print(f"Loading Data from: {data_dir}")

    # 2. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    # 3. CHECK CLASS NAMES
    class_names = train_ds.class_names
    print(f"CLASSES FOUND: {class_names}")
    # Under the new dataset structure, this will output: ['fire', 'safe']
    # 0 = FIRE, 1 = SAFE

    # 4. Standardize (0-1)
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 5. Build High-Capacity SpaceNet (Optimized for INT8 Survival)
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Block 1: Feature Extraction
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Geometry & Edges
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Texture Mapping
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting Training Pipeline...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if not os.path.exists('models'): 
        os.makedirs('models')
        
    model.save('models/spacenet.keras')
    print("✅ SpaceNet Model Saved successfully.")

if __name__ == "__main__":
    train_model()