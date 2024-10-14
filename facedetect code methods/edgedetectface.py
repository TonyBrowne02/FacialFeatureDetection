import cv2
import numpy as np

# Load the face image
face_image_path = r'myfacefeatures\jpegmini_optimized\face1.jpg'
face_image = cv2.imread(face_image_path)

# Check if the image was loaded successfully
if face_image is None:
    print("Error: Face image not found.")
    exit()

# Resize the face image to a smaller size
# Scale down the image based on a specific width (750 pixels)
scale_percent = 750 / face_image.shape[1]
width = int(face_image.shape[1] * scale_percent)
height = int(face_image.shape[0] * scale_percent)
dim = (width, height)

# Resize the image
resized_image = cv2.resize(face_image, dim, interpolation=cv2.INTER_AREA)

# Step 1: Preprocessing - Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Edge Detection - Use Canny edge detection to detect facial features
edges = cv2.Canny(blurred, 50, 150)

# Step 3: Skin Detection - Convert to HSV and filter based on skin tone ranges
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Define HSV range for skin tone
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Create a mask to detect skin regions
skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Apply the mask to the original image to extract skin regions
skin = cv2.bitwise_and(resized_image, resized_image, mask=skin_mask)

# Step 4: Feature Detection Using Template Matching
# Load the eye template image
eye_template_path = r'.\myfacefeatures\eye.jpg'
eye_template = cv2.imread(eye_template_path, 0)

if eye_template is None:
    print("Error: Eye template not found.")
    exit()

# Resize the eye template according to the scale factor
eye_template_resized = cv2.resize(eye_template, (0, 0), fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_AREA)

# Apply template matching to detect eyes in the grayscale image
result = cv2.matchTemplate(gray, eye_template_resized, cv2.TM_CCOEFF_NORMED)

# Set a threshold to select high-correlation regions
threshold = 0.5  # Adjust as needed
loc = np.where(result >= threshold)

# Debug: Print number of detected eye locations
print(f"Number of detected eye locations: {len(loc[0])}")

# Draw rectangles around detected eyes in the original resized image
for pt in zip(*loc[::-1]):  # Switch x and y coordinates
    bottom_right = (pt[0] + eye_template_resized.shape[1], pt[1] + eye_template_resized.shape[0])
    cv2.rectangle(resized_image, pt, bottom_right, (0, 255, 0), 2)  # Draw rectangle in green

# Step 5: Show the results
cv2.imshow('Original Image with Detected Eyes', resized_image)
cv2.imshow('Skin Detection', skin)
cv2.imshow('Edge Detection', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
