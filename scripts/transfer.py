import os
import shutil
import random

# Define source and destination directories
source_dir = '../data/asl'
destination_dir = '../data/asl_mini'

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Function to collect random sample of images from a letter directory
def collect_random_images(letter_dir, num_images=500):
    if os.path.isdir(letter_dir):
        # Get all images in the letter directory
        all_images = [f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select images (or all if less than num_images)
        selected_images = random.sample(all_images, min(num_images, len(all_images)))
        
        # Get the letter from the directory path
        letter = os.path.basename(letter_dir)
        dest_letter_dir = os.path.join(destination_dir, letter)
        
        # Create letter directory in destination if it doesn't exist
        os.makedirs(dest_letter_dir, exist_ok=True)
        
        # Copy selected images
        for i, image in enumerate(selected_images):
            source_path = os.path.join(letter_dir, image)
            dest_path = os.path.join(dest_letter_dir, f"{letter}_{i+1}.jpg")
            shutil.copy(source_path, dest_path)
        
        print(f"Transferred {len(selected_images)} images for letter {letter}")

# Iterate through each letter directory in the source directory
for letter in os.listdir(source_dir):
    letter_path = os.path.join(source_dir, letter)
    if os.path.isdir(letter_path):
        collect_random_images(letter_path)

print("Images collected and renamed successfully!")