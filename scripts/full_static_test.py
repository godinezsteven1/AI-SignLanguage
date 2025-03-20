import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("../models/full_asl_model.h5")

# Define class names
dataset_path = "../data/asl" #directory containing ASL dataset image directories (A, B, ..., Z, del, nothing, space)
class_names = sorted(os.listdir(dataset_path))  # Extract class names from dataset

# Load and preprocess an image
img_path = "../data/asl/A/A1.jpg"  # Update with your test image
img = image.load_img(img_path, target_size=(200, 200))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
predicted_probabilities = predictions[0]  # Get the probabilities for the first image

print(f"Predicted class: {predicted_class}")
print(f"Prediction probabilities: {predicted_probabilities}")