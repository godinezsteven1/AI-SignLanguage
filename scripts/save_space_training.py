# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Set up paths
data_dirs = ['../data/asl', '../data/dgs', '../data/lse']
preprocessed_dir = "../data/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

IMG_SIZE = 128

# Maps which letters are shared across which languages
letter_relationships = {
    "A": ["asl-dgs-lse"],
    "B": ["asl-dgs", "lse"],
    "C": ["asl-dgs-lse"],
    "D": ["asl", "dgs-lse"],
    "E": ["asl", "dgs", "lse"],
    "F": ["asl", "dgs", "lse"],
    "G": ["asl", "dgs-lse"],
    "H": ["asl", "dgs"],
    "I": ["asl-dgs-lse"],
    "J": ["asl", "dgs"],
    "K": ["asl", "dgs", "lse"],
    "L": ["asl-dgs-lse"],
    "M": ["asl-lse", "dgs"],
    "N": ["asl-lse", "dgs"],
    "O": ["asl-dgs-lse"],
    "P": ["asl", "dgs", "lse"],
    "Q": ["asl-dgs", "lse"],
    "R": ["asl-dgs-lse"],
    "S": ["asl-dgs", "lse"],
    "Sch": ["dgs"],
    "T": ["asl", "dgs", "lse"],
    "U": ["asl-dgs"],
    "V": ["asl-dgs"],
    "W": ["asl-dgs"],
    "X": ["asl-dgs"],
    "Y": ["asl-dgs"],
    "Z": ["asl", "dgs"],
}

# Helper to read and preprocess images
def get_preprocessed_images(folder_path, sample_count, padding):
    # List all files in the folder (excluding .DS_Store)
    all_files = [f for f in os.listdir(folder_path) if f != ".DS_Store"]
    
    # If the language is DGS, apply random sampling with a limit of 100 per class
    sampled_files = all_files  # No limit for ASL and LSE
    
    processed_images = []
    
    # Process each file
    for file in sampled_files:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        
        if img is None:
            continue
        
        processed_img = preprocess_image(img, padding=padding)
        
        # Only save the image if the hand was detected
        if processed_img is not None:
            processed_images.append((file, processed_img))
    
    return processed_images

# Main function
def preprocess_and_save_images():
    # Directory mapping for languages
    lang_dirs = {os.path.basename(d): d for d in data_dirs}

    # Iterate through the letter relationships to combine directories
    for letter, lang_groups in letter_relationships.items():
        for group in lang_groups:
            langs = group.split("-")
            output_class_dir = os.path.join(preprocessed_dir, f"{letter}_{group}")
            os.makedirs(output_class_dir, exist_ok=True)

            # Calculate the number of samples per language in the group
            samples_per_lang = 100 // len(langs)

            # Process each language in the group
            for lang in langs:
                lang_dir = lang_dirs[lang]
                class_path = os.path.join(lang_dir, letter)

                padding = 60 if lang == "lse" else 30
                images = get_preprocessed_images(class_path, samples_per_lang, padding)

                # Save the preprocessed images
                for file_name, img in images:
                    output_path = os.path.join(output_class_dir, f"{lang}_{file_name}")
                    cv2.imwrite(output_path, img)

def balance_class_distribution():
    # Walk through each class in the preprocessed directory
    for root, dirs, files in os.walk(preprocessed_dir):
        for class_name in dirs:
            class_path = os.path.join(root, class_name)
            image_files = [f for f in os.listdir(class_path) if f != '.DS_Store']
            
            # If there are more than 100 images, randomly sample 100
            if len(image_files) > 100:
                sampled_files = random.sample(image_files, 100)
                
                # Delete the excess images
                for file in image_files:
                    if file not in sampled_files:
                        os.remove(os.path.join(class_path, file))
                
                print(f"Class {class_name} was reduced to 100 samples.")

# Function to crop and resize images
def preprocess_image(image, padding=30):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        x_min, x_max, y_min, y_max = w, 0, h, 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
        
        # Define a minimal padding for context around the hand (20 pixels)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Calculate the width and height of the bounding box
        crop_width = x_max - x_min
        crop_height = y_max - y_min

        # Determine which dimension is larger
        if crop_width > crop_height:
            # Width is the larger dimension, so adjust based on the width
            center_y = (y_min + y_max) // 2
            half_crop = crop_width // 2
            y_min = max(0, center_y - half_crop)
            y_max = min(h, center_y + half_crop)
        else:
            # Height is the larger dimension, so adjust based on the height
            center_x = (x_min + x_max) // 2
            half_crop = crop_height // 2
            x_min = max(0, center_x - half_crop)
            x_max = min(w, center_x + half_crop)
        
        # Crop the image based on the adjusted bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Now, resize the cropped image to fit the longer side to IMG_SIZE, keeping the aspect ratio
        cropped_h, cropped_w, _ = cropped_image.shape
        scale = IMG_SIZE / max(cropped_h, cropped_w)
        new_w, new_h = int(cropped_w * scale), int(cropped_h * scale)
        resized_img = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop the image to make it square if it's not already
        if new_w > new_h:
            crop_start = (new_w - new_h) // 2
            cropped_resized_img = resized_img[:, crop_start:crop_start + new_h]
        else:
            crop_start = (new_h - new_w) // 2
            cropped_resized_img = resized_img[crop_start:crop_start + new_w, :]

        # Finally, ensure it's exactly IMG_SIZE x IMG_SIZE
        return cv2.resize(cropped_resized_img, (IMG_SIZE, IMG_SIZE))
    
    return None

# Prepare dataset by organizing class paths
def prepare_combined_dataset():
    combined_data = []
    class_weights = {}
    max_samples = 0
    
    all_classes = []
    class_counts = {}  # Dictionary to store class counts

    for root, dirs, files in os.walk(preprocessed_dir):
        for class_name in dirs:
            class_path = os.path.join(root, class_name)
            num_images = len([f for f in os.listdir(class_path) if f != ".DS_Store"])
            if num_images > 0:
                all_classes.append((class_path, class_name))
                class_counts[class_name] = num_images  # Store count

    all_classes.sort(key=lambda x: x[1])
    
    for class_path, class_name in all_classes:
        num_images = class_counts[class_name]
        max_samples = max(max_samples, num_images)

    for i, (class_path, class_name) in enumerate(all_classes):
        num_images = class_counts[class_name]
        
        # Check if class is from 'dgs' dataset and apply random sampling
        
        class_weights[i] = min(2.0, max_samples / num_images)
        combined_data.append((class_path, class_name))

    # Print out class counts
    print("\nClass Distribution:")
    for _, class_name in all_classes:
        print(f"{class_name}: {class_counts[class_name]} images")

    print("\nClass Weights:", class_weights)
    return combined_data, class_weights

# Custom Data Generator
class CombinedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, is_training=True):
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = []
        
        max_images = max(len(os.listdir(class_path)) for class_path, _ in data)
        for class_path, _ in data:
            image_files = [f for f in os.listdir(class_path) if f != '.DS_Store']
            oversample_ratio = min(2, max_images / len(image_files))
            oversampled_files = random.choices(image_files, k=int(len(image_files) * oversample_ratio))
            self.indices.extend([(class_path, f) for f in oversampled_files])
        
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return min(350, len(self.indices) // self.batch_size)  # Cap steps to 350 max
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = np.zeros((len(batch_indices), len(self.data)))
        
        for i, (class_path, image_file) in enumerate(batch_indices):
            image_path = os.path.join(class_path, image_file)
            img = cv2.imread(image_path)
            batch_x.append(img)
            
            class_idx = next(idx for idx, (path, _) in enumerate(self.data) if path == class_path)
            batch_y[i, class_idx] = 1
        
        return np.array(batch_x) / 255.0, batch_y
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

# Build Model
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
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

# Preprocess images once before training
preprocess_and_save_images()

balance_class_distribution()

# Prepare dataset
combined_data, class_weights = prepare_combined_dataset()
num_classes = len(combined_data)

# Create training and validation generators
batch_size = 64
train_generator = CombinedDataGenerator(combined_data, batch_size, is_training=True)
val_generator = CombinedDataGenerator(combined_data, batch_size, is_training=False)

# Build and compile model
model = build_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
)

# Save model
model.save('../models/full_model4.h5')