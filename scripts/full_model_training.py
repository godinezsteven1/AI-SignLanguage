# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

# Set up paths
data_dirs = ['../data/asl', '../data/dgs', '../data/lse']  # Updated to include all datasets

for data_dir in data_dirs:
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it is an image
            except Exception as e:
                print(f"Invalid image file: {file_path} - Error: {e}")

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create a function to preprocess images
def preprocess_image(image):
    # Crop the image if it's rectangular (like LSE)
    if image.shape[0] != image.shape[1]:  # Check if the image is not square
        min_dim = min(image.shape[0], image.shape[1])
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        cropped_image = image[
            center_y - min_dim // 2:center_y + min_dim // 2,
            center_x - min_dim // 2:center_x + min_dim // 2
        ]
    else:
        cropped_image = image

    # Resize to 200x200
    resized_image = cv2.resize(cropped_image, (200, 200))
    return resized_image

# Create a function to create a custom generator
def create_generators(data_dir, batch_size):
    # Create the training generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Create the validation generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Custom generator to yield preprocessed images
def custom_generator(generator):
    while True:
        x, y = next(generator)  # Use next() to get the next batch
        processed_images = []
        valid_labels = []
        
        for img, label in zip(x, y):
            try:
                processed_images.append(preprocess_image(img))  # Preprocess images
                valid_labels.append(label)  # Keep the corresponding label
            except Exception as e:
                print(f"Error processing image: {e}")  # Log the error
                continue  # Skip this image
        
        if processed_images:  # Ensure there are valid images to yield
            yield np.array(processed_images), np.array(valid_labels)

# Create generators for each dataset
train_generators = []
validation_generators = []
class_indices_list = []

for data_dir in data_dirs:
    train_gen, val_gen = create_generators(data_dir, batch_size=32)
    train_generators.append(train_gen)
    validation_generators.append(val_gen)
    # Store class indices from the training generator
    class_indices_list.append(train_gen.class_indices)

# Build a simple CNN model
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        Input(shape=(200, 200, 3)),  # Use Input layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Updated for total classes
    ])
    return model

# Train the model on all datasets
for train_gen, val_gen in zip(train_generators, validation_generators):
    num_classes = len(train_gen.class_indices)  # Get the number of classes for the current dataset
    model = build_model(num_classes)  # Build a new model for the current dataset
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size
    
    model.fit(custom_generator(train_gen), 
              validation_data=custom_generator(val_gen), 
              epochs=10, 
              steps_per_epoch=steps_per_epoch, 
              validation_steps=validation_steps)

# Save the model
model.save('../models/full_sign_language_model.h5')  # Updated model name