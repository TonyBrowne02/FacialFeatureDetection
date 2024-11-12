import cv2
import numpy as np

# Load pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the camera
cap = cv2.VideoCapture(0)

def detect_faces_with_edge_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    return edges

def detect_faces_with_skin_color(frame):
    # Convert to YCrCb color space to detect skin color
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Set skin color range in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    # Mask the skin color region
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    return skin_mask

def detect_faces_with_haar_cascade(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Edge Detection
    edges = detect_faces_with_edge_detection(frame)
    cv2.imshow('Edge Detection', edges)

    # Skin Color Detection
    skin = detect_faces_with_skin_color(frame)
    cv2.imshow('Skin Color Detection', skin)

    # Haar Cascade Detection
    haar_result = detect_faces_with_haar_cascade(frame.copy())
    cv2.imshow('Haar Cascade Detection', haar_result)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()



"""
cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video frame rate
    delay = int(1000 / fps)  # Calculate delay between frames for real-time display
"""