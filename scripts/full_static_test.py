import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load the trained model
model = load_model("../models/full_model_with_cropping.h5")

# Define class names
dataset_path = "../data"  # Update to the parent directory containing ASL and DGS datasets
class_names = []

# Loop through each language directory
for language in sorted(os.listdir(dataset_path)):
    if language == "asl_mini":  # Skip the asl directory
        continue
    language_path = os.path.join(dataset_path, language)
    if os.path.isdir(language_path):  # Check if it's a directory
        for class_name in os.listdir(language_path):
            if class_name != ".DS_Store":  # Exclude hidden files
                class_names.append(f"{language}_{class_name}")  # Append language and class name

class_names.sort()
print(class_names)

# Add preprocessing step to match training
def preprocess_image(image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hands
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

    # Resize to match training size (128x128)
    resized_image = cv2.resize(cropped_image, (128, 128))
    return resized_image

img_path = "../data/lse/E/DSC01091.JPG"
img = cv2.imread(img_path)  # Use cv2.imread instead of keras.preprocessing.image
img_array = preprocess_image(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
predicted_probabilities = predictions[0]

# Extract language from predicted class
predicted_language = predicted_class.split('_')[0]

print(f"Predicted class: {predicted_class}")
print(f"Predicted language: {predicted_language}")

# Print top 5 predictions
print("Number of classes:", len(class_names))
print("Shape of predictions:", predictions.shape)
print("Top 5 predictions:")
top_5_indices = np.argsort(predicted_probabilities)[::-1][:5]  # Get only top 5
for idx in top_5_indices:
    print(f"{class_names[idx]}: {predicted_probabilities[idx]:.6f}")