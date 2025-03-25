# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Set up paths
data_dirs = ['../data/asl_mini', '../data/dgs', '../data/lse']  # Updated to include all datasets

# Verify images and remove invalid ones
for data_dir in data_dirs:
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == '.DS_Store':  # Skip .DS_Store files
                continue
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Invalid image file: {file_path} - Error: {e}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=[0.9, 1.1],
)

""""
# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=[0.9, 1.1],
)
"""

IMG_SIZE = 128

def preprocess_image(image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hands - This is computationally expensive
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        # Get hand bounding box
        h, w, _ = image.shape
        x_min, x_max, y_min, y_max = w, 0, h, 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Crop to hand
        cropped_image = image[y_min:y_max, x_min:x_max]
    else:
        # If no hand detected, use center crop
        if image.shape[0] != image.shape[1]:
            min_dim = min(image.shape[0], image.shape[1])
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            cropped_image = image[
                center_y - min_dim // 2:center_y + min_dim // 2,
                center_x - min_dim // 2:center_x + min_dim // 2
            ]
        else:
            cropped_image = image

    # Resize to smaller size
    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    return resized_image

def prepare_combined_dataset():
    combined_data = []
    class_weights = {}
    max_samples = 0
    
    # Create a sorted list of all classes with their paths
    all_classes = []
    for data_dir in data_dirs:
        language = os.path.basename(data_dir)
        for class_name in sorted(os.listdir(data_dir)):  # Sort the class names
            if class_name == '.DS_Store':
                continue
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                full_class_name = f"{language}_{class_name}"
                all_classes.append((class_path, full_class_name))
    
    # Sort by full class name to ensure consistent ordering
    all_classes.sort(key=lambda x: x[1])
    
    # First pass: find the maximum number of samples per class
    for class_path, _ in all_classes:
        num_images = len([f for f in os.listdir(class_path) if f != '.DS_Store'])
        max_samples = max(max_samples, num_images)
    
    # Second pass: calculate weights and prepare data
    for class_path, full_class_name in all_classes:
        combined_data.append((class_path, full_class_name))
        num_images = len([f for f in os.listdir(class_path) if f != '.DS_Store'])
        class_weights[full_class_name] = max_samples / num_images
    
    return combined_data, class_weights

"""
# Build the CNN model
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        Input(shape=(200, 200, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add BatchNorm
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add BatchNorm
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Increased filters
        tf.keras.layers.BatchNormalization(),  # Add BatchNorm
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),  # Increased units
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),  # Added extra dense layer
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
"""
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # Changed to use IMG_SIZE (128)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
    
# Prepare the combined dataset
combined_data, class_weights = prepare_combined_dataset()
num_classes = len(combined_data)

# Create a custom data generator
class CombinedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, is_training=True, **kwargs):
        super().__init__(**kwargs)  # Add this line
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = []
        
        # Create indices for all images in all classes
        for class_path, _ in self.data:
            image_files = [f for f in os.listdir(class_path) if f != '.DS_Store']
            self.indices.extend([(class_path, f) for f in image_files])
        
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = np.zeros((len(batch_indices), len(self.data)))
        
        for i, (class_path, image_file) in enumerate(batch_indices):
            image_path = os.path.join(class_path, image_file)
            
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = preprocess_image(img)
            batch_x.append(img)
            
            # Create one-hot encoded label
            class_idx = next(idx for idx, (path, _) in enumerate(self.data) if path == class_path)
            batch_y[i, class_idx] = 1
        
        return np.array(batch_x) / 255.0, batch_y
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

# Create training and validation generators
batch_size = 64
train_generator = CombinedDataGenerator(combined_data, batch_size, is_training=True)
val_generator = CombinedDataGenerator(combined_data, batch_size, is_training=False)

# Build and compile the model
model = build_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Number of classes: {num_classes}")
print("Class distribution:")
for class_path, class_name in combined_data:
    num_images = len([f for f in os.listdir(class_path) if f != '.DS_Store'])
    print(f"{class_name}: {num_images} images")

# Train the model
history = model.fit(
    train_generator,
    epochs=15,  # Reduced epochs
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reduced patience
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2  # Reduced patience
        )
    ]
)

# Save the model
model.save('../models/asl_model_with_cropping.h5')