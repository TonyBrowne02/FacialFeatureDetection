import cv2

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Read an image
image = cv2.imread(r'<IMAGE FILE PATH')

# Scale down the image
scale_percent = 750 / image.shape[1]
width = int(image.shape[1] * scale_percent)
height = int(image.shape[0] * scale_percent)
dim = (width, height)

# Resize the image
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop over the detected faces
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Define the region of interest (ROI) for eye detection, focusing on the upper half of the face
    roi_gray = gray[y:y + int(h / 2), x:x + w]  # Only upper half of the face
    roi_color = resized_image[y:y + int(h / 2), x:x + w]

    # Detect eyes in the ROI
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around the eyes
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Display the output
cv2.imshow('Face and Eye Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
