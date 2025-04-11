import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
# import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

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
        # If no hand detected, return None for hand_crop
        # We still need to return a dummy preprocessed image to maintain the return structure
        dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = np.expand_dims(dummy_image, axis=0)
        return preprocessed, None
    
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

# def display_result(display, frame, accumulated_string, confidence, hand_crop=None, current_letter = ""):
#     """Display frame and recognition results"""
#     display.fill((255, 255, 255))
    
#     # Convert OpenCV's BGR to RGB for PyGame
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_rgb = cv2.resize(frame_rgb, (320, 240))
#     frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
#     display.blit(frame_surface, (20, 20))
    
#     # Display the cropped hand if available
#     if hand_crop is not None:
#         hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
#         hand_rgb = cv2.resize(hand_rgb, (128, 128))
#         hand_surface = pygame.surfarray.make_surface(hand_rgb.swapaxes(0, 1))
#         display.blit(hand_surface, (360, 20))
    
#     # Display the accumulated string
#     font = pygame.font.Font('freesansbold.ttf', 36)
#     text_surface = font.render(accumulated_string, True, (0, 0, 0))
#     text_rect = text_surface.get_rect()
#     text_rect.center = (400, 300)
#     display.blit(text_surface, text_rect)

#     # Display the current letter
#     current_letter_font = pygame.font.Font('freesansbold.ttf', 24)
#     current_letter_surface = current_letter_font.render(f"Current: {current_letter}", True, (0, 0, 255))
#     display.blit(current_letter_surface, (300, 350))
    
#     # Display confidence
#     conf_font = pygame.font.Font('freesansbold.ttf', 24)
#     conf_text = f"Confidence: {confidence:.2f}"
#     conf_surface = conf_font.render(conf_text, True, (0, 0, 0))
#     display.blit(conf_surface, (300, 400))
    
#     # Instructions
#     instructions_1 = "Press C to clear string"
#     instructions_2 = "Press B to backspace"
#     instructions_3 = "Press Q to quit"
#     inst1_surface = conf_font.render(instructions_1, True, (100, 100, 100))
#     inst2_surface = conf_font.render(instructions_2, True, (100, 100, 100))
#     inst3_surface = conf_font.render(instructions_3, True, (100, 100, 100))
#     display.blit(inst1_surface, (320, 460))
#     display.blit(inst2_surface, (320, 500))
#     display.blit(inst3_surface, (320, 540))
    
#     pygame.display.update()

def display_result(display, frame, accumulated_string, confidence, hand_crop=None, current_letter=""):
    """Display frame and recognition results based on the new layout"""
    display.fill((0, 255, 255))
    
    # Convert OpenCV's BGR to RGB for PyGame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (300, 240))  # Webcam feed on left
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    display.blit(frame_surface, (20, 20))  # Left position
    
    # Display the cropped hand if available (center top)
    if hand_crop is not None:
        hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        hand_rgb = cv2.resize(hand_rgb, (128, 128))
        hand_surface = pygame.surfarray.make_surface(hand_rgb.swapaxes(0, 1))
        display.blit(hand_surface, (340, 20))  # Center top position
    else:
        # Display empty box for hand crop
        pygame.draw.rect(display, (200, 200, 200), (340, 20, 128, 128), 2)
    
    # Display "TEXT:" label with medium size font
    label_font = pygame.font.Font('freesansbold.ttf', 32)
    text_label = label_font.render("TEXT:", True, (0, 0, 0))
    display.blit(text_label, (50, 350))
    
    # Create a section for result_text and confidence (right side) with smaller font
    result_font = pygame.font.Font('freesansbold.ttf', 22)
    
    # Display current letter and result text
    result_text_surface = result_font.render(f"Result: {current_letter}", True, (0, 0, 0))
    display.blit(result_text_surface, (340, 180))
    
    # Display confidence
    conf_text = f"Confidence: {confidence:.2f}"
    conf_surface = result_font.render(conf_text, True, (0, 0, 0))
    display.blit(conf_surface, (340, 230))
    
    # Display "TEXT:" label
    # text_label = font.render("TEXT:", True, (0, 0, 0))
    # display.blit(text_label, (50, 350))
    
    # Draw a rectangular box for the accumulated text
    pygame.draw.rect(display, (240, 240, 240), (150, 330, 550, 70), 0)  # Filled box
    pygame.draw.rect(display, (0, 0, 0), (150, 330, 550, 70), 2)  # Border
    
    # Display the accumulated string inside the box (largest font)
    text_font = pygame.font.Font('freesansbold.ttf', 32)
    text_surface = text_font.render(accumulated_string, True, (0, 0, 0))
    text_rect = text_surface.get_rect()
    text_rect.midleft = (160, 365)  # Center-left aligned in the box
    display.blit(text_surface, text_rect)
    
    # Instructions at the bottom (smallest font)
    instructions_font = pygame.font.Font('freesansbold.ttf', 18)
    instructions_1 = "Instruction 1: Press C to clear string"
    instructions_2 = "Instruction 2: Press B to backspace"
    instructions_3 = "Instruction 3: Press Q to quit"
    
    inst1_surface = instructions_font.render(instructions_1, True, (0, 0, 0))
    inst2_surface = instructions_font.render(instructions_2, True, (0, 0, 0))
    inst3_surface = instructions_font.render(instructions_3, True, (0, 0, 0))
    
    display.blit(inst1_surface, (50, 460))
    display.blit(inst2_surface, (50, 500))
    display.blit(inst3_surface, (50, 540))
    
    pygame.display.update()

def main():
    # Define mapping from class indices to sign meanings based on ASL dataset
    # This needs to be updated based on your actual classes from the model
    sign_classes = {
        0: "A (asl/dgs/lse)", 
        1: "B (asl/dgs)", 
        2: "B (lse)", 
        3: "C (asl/dgs/lse)", 
        4: "D (asl)", 
        5: "D (dgs/lse)", 
        6: "E (asl)", 
        7: "E (dgs)", 
        8: "E (lse)", 
        9: "F (asl)", 
        10: "F (dgs)", 
        11: "F (lse)", 
        12: "G (asl)", 
        13: "G (dgs/lse)", 
        14: "H (asl)", 
        15: "H (dgs)", 
        16: "I (asl/dgs/lse)", 
        17: "J (asl)", 
        18: "J (dgs)", 
        19: "K (asl)", 
        20: "K (dgs)", 
        21: "K (lse)", 
        22: "L (asl/dgs/lse)", 
        23: "M (asl/lse)", 
        24: "M (dgs)", 
        25: "N (asl/lse)", 
        26: "N (dgs)", 
        27: "O (asl/dgs/lse)", 
        28: "P (asl)", 
        29: "P (dgs)", 
        30: "P (lse)", 
        31: "Q (asl/dgs)", 
        32: "Q (lse)", 
        33: "R (asl/dgs/lse)", 
        34: "S (asl/dgs)", 
        35: "S (lse)", 
        36: "Sch (dgs)", 
        37: "T (asl)", 
        38: "T (dgs)", 
        39: "T (lse)", 
        40: "U (asl/dgs)", 
        41: "V (asl/dgs) or U (lse)", 
        42: "W (asl/dgs)", 
        43: "X (asl/dgs)", 
        44: "Y (asl/dgs)", 
        45: "Z (asl)", 
        46: "Z (dgs)"
    }
    
    # Initialize camera
    camera = initialize_camera(0)
    if camera is None:
        print("Failed to initialize camera. Exiting...")
        return
    
    # Load model
    model_path = "models\\full_model_combined_classes.h5"  # Update path as needed
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
    confidence_threshold = 0.7
    accumulated_string = ""
    last_letter = ""
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
