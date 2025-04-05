import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
import os
import time

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
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Crop to hand
        cropped_image = image[y_min:y_max, x_min:x_max]

    else:
        # If no hand detected, return None for hand_crop
        # We still need to return a dummy preprocessed image to maintain the return structure
        dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = np.expand_dims(dummy_image, axis=0)
        return preprocessed, None
    
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

# def display_result(display, frame, result_text, confidence, hand_crop=None):
def display_result(display, frame, accumulated_string, confidence, hand_crop=None, current_letter = ""):
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
    

    
    # Display the accumulated string
    font = pygame.font.Font('freesansbold.ttf', 36)
    text_surface = font.render(accumulated_string, True, (0, 0, 0))
    text_rect = text_surface.get_rect()
    text_rect.center = (400, 300)
    display.blit(text_surface, text_rect)

    # Display the current letter
    current_letter_font = pygame.font.Font('freesansbold.ttf', 24)
    # current_letter_surface = current_letter_font.render(f"Current: {result_text}", True, (0, 0, 255))
    current_letter_surface = current_letter_font.render(f"Current: {current_letter}", True, (0, 0, 255))
    display.blit(current_letter_surface, (300, 350))

    # Display confidence
    conf_font = pygame.font.Font('freesansbold.ttf', 24)
    conf_text = f"Confidence: {confidence:.2f}"
    conf_surface = conf_font.render(conf_text, True, (0, 0, 0))
    display.blit(conf_surface, (300, 400))
    
    # Instructions
    instructions_1 = "Press C to clear string"
    instructions_2 = "Press B to backspace"
    instructions_3 = "Press Q to quit"
    inst1_surface = conf_font.render(instructions_1, True, (100, 100, 100))
    inst2_surface = conf_font.render(instructions_2, True, (100, 100, 100))
    inst3_surface = conf_font.render(instructions_3, True, (100, 100, 100))
    display.blit(inst1_surface, (320, 460))
    display.blit(inst2_surface, (320, 500))
    display.blit(inst3_surface, (320, 540))
    
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
    25: "Z (ASL)", 26: "Delete (ASL)", 27: "Nothing (ASL)", 28: "Space (ASL)",
    29: "A (DGS)", 30: "B (DGS)", 31: "C (DGS)", 32: "D (DGS)", 33: "E (DGS)",
    34: "F (DGS)", 35: "G (DGS)", 36: "H (DGS)", 37: "I (DGS)", 38: "J (DGS)",
    39: "K (DGS)", 40: "L (DGS)", 41: "M (DGS)", 42: "N (DGS)", 43: "O (DGS)",
    44: "P (DGS)", 45: "Q (DGS)", 46: "R (DGS)", 47: "S (DGS)", 48: "Sch (DGS)",
    49: "T (DGS)", 50: "U (DGS)", 51: "V (DGS)", 52: "W (DGS)", 53: "X (DGS)",
    54: "Y (DGS)", 55: "Z (DGS)", 56: "A (LSE)", 57: "B (LSE)", 58: "C (LSE)",
    59: "D (LSE)", 60: "E (LSE)", 61: "F (LSE)", 62: "G (LSE)", 63: "I (LSE)",
    64: "K (LSE)", 65: "L (LSE)", 66: "M (LSE)", 67: "N (LSE)", 68: "O (LSE)",
    69: "P (LSE)", 70: "Q (LSE)", 71: "R (LSE)", 72: "S (LSE)", 73: "T (LSE)",
    74: "U (LSE)"
    }

    
    # Initialize camera
    camera = initialize_camera() #Change the device_id according to your system
    if camera is None:
        print("Failed to initialize camera. Exiting...")
        return
    
    # Load model
    model_path = "F:\\AI-SignLanguage\\models\\full_model_with_cropping.h5"  # Update path as needed
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Initialize display
    display = initialize_display()
    
    # Main processing loop
    running = True
    prev_result = ""
    confidence_threshold = 0.6

    accumulated_string = ""
    last_letter = ""
    letter_count = 0
    no_hand_count = 0

    last_prediction_time = 0
    prediction_delay = 2.0  # Delay in seconds
    is_waiting = False
    waiting_message = ""
    
    while running:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_b:
                    accumulated_string = accumulated_string[:-1] if accumulated_string else ""
                elif event.key == pygame.K_c:
                    accumulated_string = ""
        
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        # Mirror image for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Preprocess frame for model input
        processed_frame, hand_crop = preprocess_image(frame)
        
        current_time = time.time()
        
        if is_waiting and (current_time - last_prediction_time < prediction_delay):
            # Still in waiting period, display countdown
            remaining_time = round(prediction_delay - (current_time - last_prediction_time), 1)
            waiting_message = f"Wait: {remaining_time}s"
            display_result(display, frame, accumulated_string, 0.0, hand_crop, waiting_message)
            pygame.time.delay(10)
            continue
        elif is_waiting:
            # Waiting period is over
            is_waiting = False
            last_letter = ""
        
        # Run prediction only if hand is detected (if hand_crop is not None)
        if hand_crop is not None and hand_crop.size > 0:
            # Get model prediction
            prediction = model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            if confidence > confidence_threshold:
                result_text = sign_classes.get(predicted_class, f"Unknown Sign ({predicted_class})")
                # Handle special cases
                if result_text == "Space (ASL)":
                    accumulated_string += " "
                    is_waiting = True
                    last_prediction_time = current_time
                    last_letter = ""
                elif result_text == "Delete (ASL)":
                    accumulated_string = accumulated_string[:-1] if accumulated_string else ""
                    is_waiting = True
                    last_prediction_time = current_time
                    last_letter = ""
                elif not is_waiting and result_text != last_letter:
                    accumulated_string += result_text[0]
                    last_letter = result_text
                    is_waiting = True
                    last_prediction_time = current_time
                    waiting_message = f"Wait: {prediction_delay}s"

            else:
                result_text = "Uncertain"
        else:
            result_text = "No hand detected"
            confidence = 0.0
            no_hand_count += 1
            if no_hand_count > 60:
                last_letter = ""
                no_hand_count = 0
        
        # Display result
        display_result(display, frame, accumulated_string, confidence, hand_crop, waiting_message if is_waiting else result_text)
        
        # Add a small delay for better performance
        pygame.time.delay(10)
    
    # Clean up resources
    camera.release()
    hands.close()
    pygame.quit()

if __name__ == "__main__":
    main()
