import cv2

def find_working_camera(max_index=10):
    print("🔍 Searching for available cameras...")
    found = False
    for i in range(max_index):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f" Camera found at index {i}")
                cap.release()
                found = True
                break
            else:
                print(f" Camera at index {i} opened but failed to read frame.")
        else:
            print(f" Camera at index {i} could not be opened.")
        cap.release()
    if not found:
        print(" No working camera found.")

find_working_camera()
