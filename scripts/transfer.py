import os
import shutil

# Define source and destination directories
source_dir = '../data/asl'  # Update with your asl directory path
destination_dir = '../data/asl_mini'  # Update with your asl_mini directory path

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate through each letter directory in the source directory
for letter in os.listdir(source_dir):
    letter_path = os.path.join(source_dir, letter)
    
    # Check if it's a directory
    if os.path.isdir(letter_path):
        # Create a corresponding directory in the destination
        dest_letter_path = os.path.join(destination_dir, letter)
        os.makedirs(dest_letter_path, exist_ok=True)
        
        # Get all images in the letter directory
        images = sorted(os.listdir(letter_path))  # Sort to ensure consistent ordering
        
        # Copy every 100th image starting from 1
        for i in range(0, len(images), 100):
            # Calculate the index for the image to copy
            image_index = i + 1  # Adjusting to get 1-based index
            if image_index <= len(images):  # Ensure the index is within bounds
                image_to_copy = images[i]
                source_image_path = os.path.join(letter_path, image_to_copy)
                destination_image_path = os.path.join(dest_letter_path, image_to_copy)
                
                # Copy the image
                shutil.copy(source_image_path, destination_image_path)

print("Selected images copied successfully!")