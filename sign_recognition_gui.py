import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
import sys
import os
import time
from pygame.locals import *
sys.path.append(os.path.join(os.path.dirname(__file__), 'NLP'))
from pipeline import NLPpipeline
from collections import deque
 

class ASLDetectionSystem:
    def __init__(self, width=800, height=600, model_path="models/full_model_combined_classes.h5"): 
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # UI colors
        self.BLUE = (74, 122, 255)
        self.DARK_BLUE = (53, 98, 230)
        self.WHITE = (255, 255, 255)
        self.LIGHT_GRAY = (245, 245, 247)
        self.GRAY = (220, 220, 220)
        self.DARK_GRAY = (120, 120, 120)
        self.BLACK = (0, 0, 0)
        self.CYAN = (0, 255, 255)
        self.GREEN = (75, 200, 100)
        
        # Window setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Sign Language Detection System')
        
        # Fonts
        self.header_font = pygame.font.SysFont('Arial', 24)
        self.large_font = pygame.font.SysFont('Arial', 60)
        self.medium_font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 14)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                        max_num_hands=1, 
                                        min_detection_confidence=0.7)
        
        # Constants for image processing
        self.IMG_SIZE = 128
        
        # Initialize camera
        self.camera = self.initialize_camera()
        self.camera_available = self.camera is not None
        
        # Load model
        self.model = self.load_model(model_path)
        if self.model is None:
            print("Failed to load model. Exiting...")
            sys.exit()
        
        # Recognition variables
        self.current_detection = ""
        self.confidence = 0.0
        self.accumulated_string = ""

        # Hand positioning delay variables
        self.hand_detected = False
        self.hand_detection_time = 0
        self.hand_positioning_delay = 2.0  # 2 second delay for positioning hand
        self.ready_for_detection = False
        
        # State variables
        self.is_waiting = False
        self.last_prediction_time = 0
        self.prediction_delay = 2.0
        self.last_letter = ""
        self.waiting_message = ""
        self.no_hand_count = 0
        self.confidence_threshold = 0.7

        # NLP variables
        self.confidence_counter = {
            "asl" : 0.00,
            "dgs" : 0.00,
            "lse" : 0.00
        }
        self.iso = {
            "asl" : "en",
            "dgs" : "de",
            "lse" : "es"
        }
        self.probable_language = None
        self.isoCode = "en"
        self.lang_history = deque(maxlen=10)
        
        # Main clock
        self.clock = pygame.time.Clock()
        
        # Mapping from class indices to sign meanings
        self.sign_classes = {
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
    
    def initialize_camera(self, device_id=0):
        """Initialize and connect to the webcam"""
        try:
            cam = cv2.VideoCapture(device_id)
            if not cam.isOpened():
                print(f"Error: Unable to access camera with device ID {device_id}")
                return None
                
            # Set camera properties
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return cam
        except Exception as e:
            print(f"Camera error: {e}")
            return None
    
    def load_model(self, model_path):
        """Load the trained CNN model for hand sign recognition"""
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image using the same method as in the training code"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_image)
        
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
            
            # Store landmarks for drawing
            self.hand_landmarks = results.multi_hand_landmarks
            
            # Hand is detected
            if not self.hand_detected:
                self.hand_detected = True
                self.hand_detection_time = time.time()
                self.ready_for_detection = False
            
        else:
            # If no hand detected, return a dummy preprocessed image to maintain the return structure
            dummy_image = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3))
            preprocessed = np.expand_dims(dummy_image, axis=0)
            self.hand_landmarks = None
            
            # Reset hand detection state
            self.hand_detected = False
            self.ready_for_detection = False
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
        resized_image = cv2.resize(cropped_image, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Normalize
        normalized_image = resized_image / 255.0
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized_image, axis=0)
        
        return preprocessed, cropped_image
    
    def draw_rounded_rect(self, surface, rect, color, radius=0.4):
        """Draw a rounded rectangle"""
        rect = pygame.Rect(rect)
        color = pygame.Color(*color)
        alpha = color.a
        color.a = 0
        pos = rect.topleft
        rect.topleft = 0, 0
        rectangle = pygame.Surface(rect.size, pygame.SRCALPHA)
        
        circle = pygame.Surface([min(rect.size)*3]*2, pygame.SRCALPHA)
        pygame.draw.ellipse(circle, (0, 0, 0), circle.get_rect(), 0)
        circle = pygame.transform.smoothscale(circle, [int(min(rect.size)*radius)]*2)
        
        radius = rectangle.blit(circle, (0, 0))
        radius.bottomright = rect.bottomright
        rectangle.blit(circle, radius)
        radius.topright = rect.topright
        rectangle.blit(circle, radius)
        radius.bottomleft = rect.bottomleft
        rectangle.blit(circle, radius)
        
        rectangle.fill((0, 0, 0), rect.inflate(-radius.w, 0))
        rectangle.fill((0, 0, 0), rect.inflate(0, -radius.h))
        
        rectangle.fill(color, special_flags=pygame.BLEND_RGBA_MAX)
        rectangle.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MIN)
        
        surface.blit(rectangle, pos)
    
    def draw_header(self):
        """Draw the header section"""
        # Header background
        self.draw_rounded_rect(self.screen, (20, 20, 760, 60), self.BLUE, 0.3)
        
        # App title
        title = self.header_font.render('Sign Language Detection System', True, self.WHITE)
        self.screen.blit(title, (40, 36))

        # Instructions
        instructions = self.small_font.render('Press Enter to see NLP result, Press C to clear, B to backspace, Q to quit', True, self.DARK_GRAY)
        self.screen.blit(instructions, (40, 570))
    
    def draw_camera_feed(self, frame):
        """Draw the camera feed section"""
        # Camera panel background
        self.draw_rounded_rect(self.screen, (20, 100, 500, 375), self.WHITE, 0.1)
        self.draw_rounded_rect(self.screen, (40, 120, 460, 335), self.LIGHT_GRAY, 0.1)
        
        # If camera is available, display feed
        if self.camera_available and frame is not None:
            try:
                # Convert OpenCV BGR to RGB for pygame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (460, 335))
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                self.screen.blit(frame_surface, (40, 120))
                
                # Draw hand landmarks if available with status-based color
                if self.hand_landmarks:
                    landmark_color = self.GREEN if self.ready_for_detection else self.CYAN
                    for landmarks in self.hand_landmarks:
                        for landmark in landmarks.landmark:
                            x = int(landmark.x * 460) + 40
                            y = int(landmark.y * 335) + 120
                            pygame.draw.circle(self.screen, landmark_color, (x, y), 3, 0)
                
                # Display positioning instructions or status
                if self.hand_detected and not self.ready_for_detection:
                    # Calculate remaining time
                    elapsed = time.time() - self.hand_detection_time
                    remaining = max(0, self.hand_positioning_delay - elapsed)
                    status_text = f"Hold position: {remaining:.1f}s"
                    status_color = self.CYAN
                elif self.ready_for_detection:
                    status_text = "Ready for detection!"
                    status_color = self.GREEN
                else:
                    status_text = "Position your hand"
                    status_color = self.DARK_GRAY
            except Exception as e:
                print(f"Error displaying camera feed: {e}")
                # Fallback to placeholder
                placeholder_text = self.medium_font.render('Camera Feed', True, self.DARK_GRAY)
                text_rect = placeholder_text.get_rect(center=(270, 287))
                self.screen.blit(placeholder_text, text_rect)
                # Display the status text
                status_surface = self.medium_font.render(status_text, True, status_color)
                status_rect = status_surface.get_rect(center=(270, 450))
                self.screen.blit(status_surface, status_rect)
                instruction_text = self.small_font.render('Position your hand in the circle', True, self.DARK_GRAY)
                instruction_rect = instruction_text.get_rect(center=(270, 410))
                self.screen.blit(instruction_text, instruction_rect)
        else:
            # Display placeholder if no camera
            placeholder_text = self.medium_font.render('Camera Feed', True, self.DARK_GRAY)
            text_rect = placeholder_text.get_rect(center=(270, 287))
            self.screen.blit(placeholder_text, text_rect)
            
            instruction_text = self.small_font.render('Position your hand in the circle', True, self.DARK_GRAY)
            instruction_rect = instruction_text.get_rect(center=(270, 410))
            self.screen.blit(instruction_text, instruction_rect)
            
            # Circle guide
            pygame.draw.circle(self.screen, self.DARK_GRAY, (270, 287), 100, 2)
            pygame.draw.circle(self.screen, self.DARK_GRAY, (270, 287), 100, 1)
    
    def draw_hand_crop(self, hand_crop):
        """Draw the hand crop section"""
        # Hand crop panel background
        self.draw_rounded_rect(self.screen, (540, 100, 240, 170), self.WHITE, 0.1)
        
        # Title
        crop_title = self.small_font.render('Hand Crop:', True, self.BLACK)
        self.screen.blit(crop_title, (560, 110))
        
        # Display the hand crop if available
        if hand_crop is not None:
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_rgb = cv2.resize(hand_rgb, (128, 128))
            hand_surface = pygame.surfarray.make_surface(hand_rgb.swapaxes(0, 1))
            
            # Center the hand crop in the panel
            crop_x = 560 + (200 - 128) // 2
            crop_y = 135
            self.screen.blit(hand_surface, (crop_x, crop_y))
        else:
            # Show empty square if no hand detected
            empty_rect = pygame.Rect(560 + (200 - 128) // 2, 135, 128, 128)
            pygame.draw.rect(self.screen, self.GRAY, empty_rect, 2)
            
            no_hand_text = self.small_font.render('No hand detected', True, self.DARK_GRAY)
            text_rect = no_hand_text.get_rect(center=(660, 199))
            self.screen.blit(no_hand_text, text_rect)
    
    def draw_detection_results(self):
        """Draw the detection results panel"""
        # Results panel background
        self.draw_rounded_rect(self.screen, (540, 290, 240, 185), self.WHITE, 0.1)
        
        # Title
        detection_title = self.small_font.render('Detected Sign:', True, self.BLACK)
        self.screen.blit(detection_title, (560, 310))
        
        # Current detection (letter)
        display_letter = self.current_detection[0] if self.current_detection and self.current_detection != "No hand detected" and self.current_detection != "Uncertain" else "?"
        letter = self.large_font.render(display_letter, True, self.BLUE)
        letter_rect = letter.get_rect(center=(660, 360))
        self.screen.blit(letter, letter_rect)
        
        # Full detection text
        if self.current_detection:
            full_text = self.small_font.render(self.current_detection, True, self.BLACK)
            full_rect = full_text.get_rect(center=(660, 410))
            self.screen.blit(full_text, full_rect)
        
        # Confidence score
        conf_percentage = int(self.confidence * 100) if self.confidence else 0
        confidence_text = self.small_font.render(f'Confidence: {conf_percentage}%', True, self.BLACK)
        self.screen.blit(confidence_text, (560, 440))
        
        # Display waiting message if applicable
        if self.is_waiting:
            waiting_text = self.small_font.render(self.waiting_message, True, self.DARK_BLUE)
            self.screen.blit(waiting_text, (560, 460))
    
    def draw_accumulated_string_panel(self):
        """Draw the history/sequence panel"""
        # History panel background
        self.draw_rounded_rect(self.screen, (20, 495, 760, 85), self.WHITE, 0.1)
        
        # Title
        history_title = self.small_font.render('Text:', True, self.BLACK)
        self.screen.blit(history_title, (40, 510))
        
        # Draw accumulated string
        text_rect = pygame.Rect(40, 530, 600, 40)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, text_rect, 0)
        pygame.draw.rect(self.screen, self.GRAY, text_rect, 1)
        
        # Display text with larger font
        text_font = pygame.font.SysFont('Arial', 24)
        text_surface = text_font.render(self.accumulated_string, True, self.BLACK)
        text_pos_rect = text_surface.get_rect(midleft=(50, 550))
        self.screen.blit(text_surface, text_pos_rect)
        
        # Cursor blinking animation
        seconds = pygame.time.get_ticks() / 1000
        if seconds % 1 > 0.5:  # Simple blink effect
            cursor_x = text_pos_rect.right + 5
            pygame.draw.rect(self.screen, self.BLACK, (cursor_x, 535, 2, 30))
        
        # Action buttons
        self.draw_rounded_rect(self.screen, (650, 530, 60, 40), self.BLUE, 0.5)
        clear_text = self.small_font.render('Clear', True, self.WHITE)
        clear_rect = clear_text.get_rect(center=(680, 550))
        self.screen.blit(clear_text, clear_rect)
        
        self.draw_rounded_rect(self.screen, (720, 530, 40, 40), self.BLUE, 0.5)
        back_text = self.small_font.render('←', True, self.WHITE)
        back_rect = back_text.get_rect(center=(740, 550))
        self.screen.blit(back_text, back_rect)
        
        self.draw_rounded_rect(self.screen, (580, 530, 60, 40), self.GREEN, 0.5)
        enter_text = self.small_font.render('Enter', True, self.WHITE)
        enter_rect = enter_text.get_rect(center=(610, 550))
        self.screen.blit(enter_text, enter_rect)
        
        # Instructions
        instructions = self.small_font.render('Press Enter to see NLP result, Press C to clear, B to backspace, Q to quit', True, self.DARK_GRAY)
        self.screen.blit(instructions, (40, 570))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.cleanup()
                return False
            
            elif event.type == MOUSEBUTTONDOWN:
                
                # Check if clear button is clicked
                if pygame.Rect(650, 530, 60, 40).collidepoint(event.pos):
                    self.accumulated_string = ""
                    # Reset the Confidence Counter
                    self.confidence_counter["asl"] = 0
                    self.confidence_counter["dgs"] = 0
                    self.confidence_counter["lse"] = 0
                    # Reset ISO to default
                    self.isoCode = "en"
                
                # Check if backspace button is clicked
                if pygame.Rect(720, 530, 40, 40).collidepoint(event.pos):
                    self.accumulated_string = self.accumulated_string[:-1] if self.accumulated_string else ""
                    # Update the Confidence Counter
                    self.confidence_counter = self.update_count(self.confidence_counter)
                    self.lang_history.pop()

                # Check if enter button is clicked (new)
                if pygame.Rect(580, 530, 60, 40).collidepoint(event.pos):
                    if self.accumulated_string != "":
                        self.probable_language = max(self.confidence_counter, key=self.confidence_counter.get)
                        self.isoCode = self.iso.get(self.probable_language)
                        text = NLPpipeline(self.accumulated_string, self.isoCode)
                        if text != self.accumulated_string:
                            self.accumulated_string = text
                        else:
                            self.accumulated_string = "Error: Could not find the words. Press C to Clear"
                        # Reset the Confidence Counter
                        self.confidence_counter["asl"] = 0
                        self.confidence_counter["dgs"] = 0
                        self.confidence_counter["lse"] = 0
                        # Reset ISO to default
                        self.isoCode = "en"
            
            elif event.type == KEYDOWN:
                # Clear text with 'c'
                if event.key == K_c:
                    self.accumulated_string = ""
                    # Reset the Confidence Counter
                    self.confidence_counter["asl"] = 0
                    self.confidence_counter["dgs"] = 0
                    self.confidence_counter["lse"] = 0
                
                # Backspace with 'b'
                elif event.key == K_b:
                    self.accumulated_string = self.accumulated_string[:-1] if self.accumulated_string else ""
                    # Update the Confidence Counter
                    self.confidence_counter = self.update_count(self.confidence_counter)
                    self.lang_history.pop()

                elif event.key == K_RETURN:
                    if self.accumulated_string != "":
                        self.probable_language = max(self.confidence_counter, key= self.confidence_counter.get)
                        self.isoCode = self.iso.get(self.probable_language)
                        text = NLPpipeline(self.accumulated_string, self.isoCode)
                        if text != self.accumulated_string:
                            self.accumulated_string = text
                        else:
                            self.accumulated_string = "Error: Could not find the words. Press C to CLear"
                        # Reset the Confidence Counter
                        self.confidence_counter["asl"] = 0
                        self.confidence_counter["dgs"] = 0
                        self.confidence_counter["lse"] = 0
                        # Reset ISO to default
                        self.isoCode = "en"
                
                # Exit with Escape or 'q'
                elif event.key == K_ESCAPE or event.key == K_q:
                    self.cleanup()
                    return False
        
        return True
    
    def update_count(self, dict_counter):
        """Update the Confidence Counter when Backspace is used"""
        key_1 = "asl"
        key_2 = "dgs"
        key_3 = "lse"
        idx = len(self.lang_history) - 1
        lang = self.lang_history[idx]
        if '/' not in lang:
            if key_1 in lang:
                dict_counter[key_1] -= 1
            if key_2 in lang:
                dict_counter[key_2] -= 1
            if key_3 in lang:
                dict_counter[key_3] -= 1
        else:
            languages = lang.split("/")
            for i in languages:
                if key_1 in languages:
                    dict_counter[key_1] -= 1/len(languages)
                if key_2 in languages:
                    dict_counter[key_2] -= 1/len(languages)
                if key_3 in languages:
                    dict_counter[key_3] -= 1/len(languages)

        return dict_counter

    def extract_and_count(self, prediction, dict_counter):
        """Extract Languages from Class names and increment the confidence counter"""
        start = None
        depth = 0
        key_1 = "asl"
        key_2 = "dgs"
        key_3 = "lse"
        text_in_paranthesis = None
        for i, char in enumerate(prediction):
            if char == '(':
                if depth == 0:
                    start = i +1
                depth +=1
            elif char == ')':
                depth -= 1
                if depth == 0 and start is not None:
                    text_in_paranthesis = prediction[start:i] 
        
        self.lang_history.append(text_in_paranthesis)

        if '/' not in text_in_paranthesis:
            if key_1 in text_in_paranthesis:
                dict_counter[key_1] += 1
            if key_2 in text_in_paranthesis:
                dict_counter[key_2] += 1
            if key_3 in text_in_paranthesis:
                dict_counter[key_3] += 1
        else:
            languages = text_in_paranthesis.split("/")
            for i in languages:
                if key_1 in languages:
                    dict_counter[key_1] += 1/len(languages)
                if key_2 in languages:
                    dict_counter[key_2] += 1/len(languages)
                if key_3 in languages:
                    dict_counter[key_3] += 1/len(languages)

        return dict_counter

    def process_frame(self, frame):
        """Process a camera frame and update detection"""
        # Preprocess frame for model input
        processed_frame, hand_crop = self.preprocess_image(frame)
        
        current_time = time.time()
        
        # Update ready_for_detection state
        if self.hand_detected and not self.ready_for_detection:
            if current_time - self.hand_detection_time >= self.hand_positioning_delay:
                self.ready_for_detection = True
        
        # Check if we're in cooldown period after a detection
        if self.is_waiting and (current_time - self.last_prediction_time < self.prediction_delay):
            remaining_time = round(self.prediction_delay - (current_time - self.last_prediction_time), 1)
            self.waiting_message = f"Wait: {remaining_time}s"
            return hand_crop
        elif self.is_waiting:
            # Cooldown period is over
            self.is_waiting = False
            self.last_letter = ""
        

        # Run prediction only if hand is detected AND ready for detection
        if hand_crop is not None and hand_crop.size > 0 and self.ready_for_detection:
            # Get model prediction
            prediction = self.model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(prediction[0])
            self.confidence = prediction[0][predicted_class]
            
            if self.confidence > self.confidence_threshold:
                self.current_detection = self.sign_classes.get(predicted_class, f"Unknown Sign ({predicted_class})")
                
                # Handle special cases
                if self.current_detection == "Space (ASL)":
                    self.accumulated_string += " "
                    self.is_waiting = True
                    self.last_prediction_time = current_time
                    self.last_letter = ""
                elif self.current_detection == "Delete (ASL)":
                    self.accumulated_string = self.accumulated_string[:-1] if self.accumulated_string else ""
                    self.is_waiting = True
                    self.last_prediction_time = current_time
                    self.last_letter = ""
                elif not self.is_waiting and self.current_detection != self.last_letter:
                    self.confidence_counter = self.extract_and_count(self.current_detection ,self.confidence_counter)
                    self.accumulated_string += self.current_detection[0]
                    self.last_letter = self.current_detection
                    self.is_waiting = True
                    self.last_prediction_time = current_time
                    self.waiting_message = f"Wait: {self.prediction_delay}s"

            else:
                self.current_detection = "Uncertain"
        elif hand_crop is not None and hand_crop.size > 0:
            # Hand is detected but not ready for detection yet
            self.current_detection = "Positioning hand..."
            self.confidence = 0.0
        else:
            self.current_detection = "No hand detected"
            self.confidence = 0.0
            self.no_hand_count += 1
            if self.no_hand_count > 60:
                self.last_letter = ""
                self.no_hand_count = 0
        
        return hand_crop
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera_available:
            self.camera.release()
        self.hands.close()
        pygame.quit()
    
    def run(self):
        """Main application loop"""
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Capture frame if camera is available
            frame = None
            hand_crop = None
            if self.camera_available:
                ret, frame = self.camera.read()
                if ret:
                    # Mirror image for more intuitive interaction
                    frame = cv2.flip(frame, 1)
                    # Process frame
                    hand_crop = self.process_frame(frame)
            
            # Clear screen
            self.screen.fill(self.LIGHT_GRAY)
            
            # Draw UI components
            self.draw_header()
            self.draw_camera_feed(frame)
            self.draw_hand_crop(hand_crop)
            self.draw_detection_results()
            self.draw_accumulated_string_panel()
            
            # Update display
            pygame.display.flip()
            
            # Cap at 30 FPS
            self.clock.tick(30)

# Main execution
if __name__ == "__main__":
    app = ASLDetectionSystem()
    app.run()
