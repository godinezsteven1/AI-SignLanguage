import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Finger tip IDs
finger_tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
finger_base_ids = [2, 5, 9, 13, 17]  # corresponding base joints

def count_fingers(hand_landmarks):
    """Count the number of extended fingers"""
    count = 0
    
    # Get hand landmark coordinates
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    # Check thumb (special case due to different bending direction)
    if landmarks[4][0] < landmarks[3][0]:  # For right hand
        count += 1
        
    # Check other fingers
    for id in range(1, 5):
        if landmarks[finger_tip_ids[id]][1] < landmarks[finger_base_ids[id]][1]:
            count += 1
            
    return count

def detect_gesture(hand_landmarks):
    """Detect simple hand gestures"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    # Detect thumbs up (thumb is extended, other fingers are closed)
    fingers = count_fingers(hand_landmarks)
    thumb_extended = landmarks[4][0] < landmarks[3][0]  # For right hand
    
    if fingers == 1 and thumb_extended:
        return "Thumbs Up"
    
    # Detect peace sign (index and middle fingers extended, others closed)
    index_extended = landmarks[8][1] < landmarks[5][1]
    middle_extended = landmarks[12][1] < landmarks[9][1]
    ring_closed = landmarks[16][1] > landmarks[13][1]
    pinky_closed = landmarks[20][1] > landmarks[17][1]
    
    if index_extended and middle_extended and ring_closed and pinky_closed:
        return "Peace Sign"
    
    # Detect open palm (all fingers extended)
    if fingers >= 4:
        return "Open Palm"
    
    # Detect pointing (only index finger extended)
    if (fingers == 1 and not thumb_extended and 
        landmarks[8][1] < landmarks[5][1]):
        return "Pointing"
    
    # Detect pinch (thumb and index finger close to each other)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                         (thumb_tip[1] - index_tip[1])**2)
    
    if distance < 0.05:
        return "Pinch"
    
    return "No Gesture"

def get_hand_center(hand_landmarks):
    """Calculate the center point of the hand"""
    x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
    y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
    
    x_center = sum(x_coordinates) / len(x_coordinates)
    y_center = sum(y_coordinates) / len(y_coordinates)
    
    return (int(x_center * frame_width), int(y_center * frame_height))

# Start capturing video
cap = cv2.VideoCapture(0)

# Get frame dimensions for UI placement
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# For FPS calculation
prev_frame_time = 0
new_frame_time = 0
import time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB and process with MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    
    # Draw FPS counter
    cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create rectangle for instructions
    cv2.rectangle(frame, (10, frame_height-140), (300, frame_height-10), (0, 0, 0), -1)
    cv2.putText(frame, "Controls:", (20, frame_height-110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "- Press 'q' to quit", (20, frame_height-80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "- Try different gestures", (20, frame_height-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks with different style
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Determine handedness
            handedness = results.multi_handedness[hand_no].classification[0].label
            
            # Count extended fingers
            finger_count = count_fingers(hand_landmarks)
            
            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Get hand center for text placement
            hand_center = get_hand_center(hand_landmarks)
            
            # Create background for text
            text_size = cv2.getTextSize(f"{handedness}: {gesture}", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                         (hand_center[0] - 10, hand_center[1] - text_size[1] - 20),
                         (hand_center[0] + text_size[0] + 10, hand_center[1] + 10),
                         (0, 0, 0), -1)
            
            # Draw info text
            cv2.putText(frame, f"{handedness}: {gesture}", 
                       (hand_center[0] - 5, hand_center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display finger count
            cv2.putText(frame, f"Fingers: {finger_count}", 
                       (hand_center[0] - 5, hand_center[1] - text_size[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # If no hands detected, show message
        cv2.putText(frame, "No hands detected", (frame_width//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow("Enhanced Hand Tracking", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()