import os
import cv2
import time
import string

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# All lowercase letters a-z (26 total)
class_labels = list(string.ascii_lowercase)  # ['a', 'b', ..., 'z']
dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera. Exiting.")
    exit()

for label in class_labels:
    class_path = os.path.join(DATA_DIR, label)
    os.makedirs(class_path, exist_ok=True)

    print(f"📸 Collecting data for class '{label.upper()}'")

    # Wait for user to press Q
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame in preview.")
            continue

        cv2.putText(frame, f'Collecting for "{label.upper()}". Press Q to start.',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print("⏳ Starting image capture in 3 seconds...")
    time.sleep(3)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        filename = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
