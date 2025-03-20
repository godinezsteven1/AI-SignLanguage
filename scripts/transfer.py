import os
import shutil

# Define source and destination directories
source_dir = '../data/dgs'  # Base directory containing user directories
destination_dir = '../data/dge_test'  # New directory for collected images

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Function to collect images from a user directory
def collect_images(user_dir, letter, start_index):
    letter_path = os.path.join(user_dir, letter)
    if os.path.isdir(letter_path):
        images = sorted(os.listdir(letter_path))  # Sort to ensure consistent ordering
        for i, image in enumerate(images):
            # Calculate the new index for the image to copy
            new_index = start_index + i + 1  # Start from the given index
            new_image_name = f"{letter}_{new_index}.jpg"  # New image name
            source_image_path = os.path.join(letter_path, image)
            destination_image_path = os.path.join(destination_dir, letter, new_image_name)

            # Create letter directory in destination if it doesn't exist
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)

            # Copy the image
            shutil.copy(source_image_path, destination_image_path)

# Dictionary to keep track of the starting index for each letter
letter_indices = {}

# Iterate through each user directory in the source directory
for user in os.listdir(source_dir):
    user_path = os.path.join(source_dir, user)
    if os.path.isdir(user_path):  # Check if it's a directory
        for letter in os.listdir(user_path):  # Iterate through each letter directory
            # Initialize the starting index for the letter if not already done
            if letter not in letter_indices:
                letter_indices[letter] = 0
            
            collect_images(user_path, letter, letter_indices[letter])  # Collect images for each letter
            
            # Update the starting index for the next user
            letter_indices[letter] += 50  # Assuming each user has 50 images per letter

print("Images collected and renamed successfully!")