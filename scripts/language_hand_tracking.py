import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load the trained model
model = load_model('../models/asl_model.h5')

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'delete', 'space', 'nothing']

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the hand region
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            
            # Ensure the bounding box is valid
            if x_min < 0: x_min = 0
            if x_max > frame.shape[1]: x_max = frame.shape[1]
            if y_min < 0: y_min = 0
            if y_max > frame.shape[0]: y_max = frame.shape[0]

            hand_region = frame[y_min:y_max, x_min:x_max]

            # Check if hand_region is not empty
            if hand_region.size > 0:
                # Preprocess the hand region for prediction
                hand_region = cv2.resize(hand_region, (200, 200))  # Resize to match model input
                hand_region = hand_region / 255.0  # Normalize
                hand_region = np.expand_dims(hand_region, axis=0)  # Add batch dimension

                # Make prediction
                predictions = model.predict(hand_region)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_letter = class_labels[predicted_class]  # Use the defined class labels

                # Display the predicted letter on the frame
                cv2.putText(frame, predicted_letter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
