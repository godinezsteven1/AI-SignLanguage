import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Constants
IMG_SIZE = 128  # Same as in the model training code

def initialize_camera(device_id=0):
    """Initialize and connect to the webcam"""
    cam = cv2.VideoCapture(device_id)
    
    if not cam.isOpened():
        print(f"Error: Unable to access camera with device ID {device_id}")
        return None
        
    # Set camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cam

def load_model(model_path):
    """Load the trained CNN model for hand sign recognition"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image using the same method as in the training code
    """
    # Convert BGR to RGB for MediaPipe
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
        padding = 30
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        crop_width = x_max - x_min
        crop_height = y_max - y_min

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
    
    # Resize to square with aspect ratio preserved
    h, w, _ = cropped_image.shape
    if h != w:
        target_size = max(h, w)
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        x_offset = (target_size - w) // 2
        y_offset = (target_size - h) // 2
        padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image
        cropped_image = padded_image
    
    # Resize to model input size
    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    normalized_image = resized_image / 255.0
    
    # Add batch dimension
    preprocessed = np.expand_dims(normalized_image, axis=0)
    
    return preprocessed, cropped_image

def initialize_display(width=800, height=600):
    """Initialize PyGame display for showing results"""
    pygame.init()
    pygame.display.set_caption('ASL Hand Sign Recognition')
    display = pygame.display.set_mode((width, height))
    return display

def display_result(display, frame, result_text, confidence, hand_crop=None):
    """Display frame and recognition results"""
    display.fill((255, 255, 255))
    
    # Convert OpenCV's BGR to RGB for PyGame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (320, 240))
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    display.blit(frame_surface, (20, 20))
    
    # Display the cropped hand if available
    if hand_crop is not None:
        hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        hand_rgb = cv2.resize(hand_rgb, (128, 128))
        hand_surface = pygame.surfarray.make_surface(hand_rgb.swapaxes(0, 1))
        display.blit(hand_surface, (360, 20))
    
    # Display recognition result
    font = pygame.font.Font('freesansbold.ttf', 36)
    text_surface = font.render(result_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect()
    text_rect.center = (400, 300)
    display.blit(text_surface, text_rect)
    
    # Display confidence
    conf_font = pygame.font.Font('freesansbold.ttf', 24)
    conf_text = f"Confidence: {confidence:.2f}"
    conf_surface = conf_font.render(conf_text, True, (0, 0, 0))
    display.blit(conf_surface, (300, 350))
    
    # Instructions
    instructions = "Press Q to quit"
    inst_surface = conf_font.render(instructions, True, (100, 100, 100))
    display.blit(inst_surface, (320, 550))
    
    pygame.display.update()

def main():
    # Define mapping from class indices to sign meanings based on ASL dataset
    # This needs to be updated based on your actual classes from the model

    sign_classes = {
    0: "A (ASL)", 1: "B (ASL)", 2: "C (ASL)", 3: "D (ASL)", 4: "E (ASL)",
    5: "F (ASL)", 6: "G (ASL)", 7: "H (ASL)", 8: "I (ASL)", 9: "J (ASL)",
    10: "K (ASL)", 11: "L (ASL)", 12: "M (ASL)", 13: "N (ASL)", 14: "O (ASL)",
    15: "P (ASL)", 16: "Q (ASL)", 17: "R (ASL)", 18: "S (ASL)", 19: "T (ASL)",
    20: "U (ASL)", 21: "V (ASL)", 22: "W (ASL)", 23: "X (ASL)", 24: "Y (ASL)",
    25: "Z (ASL)", 26: "Space (ASL)",
    27: "A (DGS)", 28: "B (DGS)", 29: "C (DGS)", 30: "D (DGS)", 31: "E (DGS)",
    32: "F (DGS)", 33: "G (DGS)", 34: "H (DGS)", 35: "I (DGS)", 36: "J (DGS)",
    37: "K (DGS)", 38: "L (DGS)", 39: "M (DGS)", 40: "N (DGS)", 41: "O (DGS)",
    42: "P (DGS)", 43: "Q (DGS)", 44: "R (DGS)", 45: "S (DGS)", 46: "Sch (DGS)",
    47: "T (DGS)", 48: "U (DGS)", 49: "V (DGS)", 50: "W (DGS)", 51: "X (DGS)",
    52: "Y (DGS)", 53: "Z (DGS)", 54: "A (LSE)", 55: "B (LSE)", 56: "C (LSE)",
    57: "D (LSE)", 58: "E (LSE)", 59: "F (LSE)", 60: "G (LSE)", 61: "I (LSE)",
    62: "K (LSE)", 63: "L (LSE)", 64: "M (LSE)", 65: "N (LSE)", 66: "O (LSE)",
    67: "P (LSE)", 68: "Q (LSE)", 69: "R (LSE)", 70: "S (LSE)", 71: "T (LSE)",
    72: "U (LSE)"
    }
    
    # Initialize camera
    camera = initialize_camera(0)
    if camera is None:
        print("Failed to initialize camera. Exiting...")
        return
    
    # Load model
    model_path = "../models/full_model3.h5"  # Update path as needed
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting...")
        return
    print("Loaded model successfully. Class names:")
    for index, sign in sign_classes.items():
        print(f"{index}: {sign}")
    # Initialize display
    display = initialize_display()
    
    # Main processing loop
    running = True
    prev_result = ""
    confidence_threshold = 0.1
    
    while running:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        # Mirror image for more intuitive interaction
        #frame = cv2.flip(frame, 1)
        
        # Preprocess frame for model input
        processed_frame, hand_crop = preprocess_image(frame)
        
        # Run prediction only if hand is detected (if hand_crop is not None)
        if hand_crop is not None and hand_crop.size > 0:
            # Get model prediction
            prediction = model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            if confidence > confidence_threshold:
                result_text = sign_classes.get(predicted_class, f"Unknown Sign ({predicted_class})")
            else:
                result_text = "Uncertain"
        else:
            result_text = "No hand detected"
            confidence = 0.3
        
        # Display result
        display_result(display, frame, result_text, confidence, hand_crop)
        
        # Add a small delay for better performance
        pygame.time.delay(10)
    
    # Clean up resources
    camera.release()
    hands.close()
    pygame.quit()

if __name__ == "__main__":
    main()
