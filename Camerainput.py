import cv2
import os
import datetime

save_directory = 'camerainput screenshots'

# Ensure the directory exists, if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        screenshot_filename = f"screenshot_{timestamp}.png"
        screenshot_path = os.path.join(save_directory, screenshot_filename)
        cv2.imwrite(screenshot_path, frame)  # Save the current frame with strokes
        print(f"Screenshot saved as {timestamp}")

vc.release()
cv2.destroyWindow("preview")