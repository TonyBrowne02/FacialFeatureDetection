from tkinter import filedialog
import cv2
import easygui as eg
import numpy as np


def image_file_explore():
    # Prompt user to select an image file
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        eg.msgbox("No File Selected.", "Error")
    return file_path


def video_file_explore():
    # Prompt user to select a video file
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if not file_path:
        eg.msgbox("No File Selected.", "Error")
        return None
    return file_path


def detect_faces_with_edge_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    return edges


def detect_faces_with_skin_mask(frame):
    # Convert to YCrCb color space to detect skin color
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Define skin color range in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    # Create a mask for skin-colored regions
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    return skin_mask


def detect_faces_with_skin_region(frame):
    # Generate a mask for the skin-colored region
    skin_mask = detect_faces_with_skin_mask(frame)
    # Extract the skin region from the original image
    skin_region = cv2.bitwise_and(frame, frame, mask=skin_mask)
    return skin_region


def capture_fixed_roi(frame):
    # Define an ROI in the center of the image
    width_ratio = 0.3
    height_ratio = 0.3
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Define ROI size based on the specified ratios
    roi_width = int(w * width_ratio)
    roi_height = int(h * height_ratio)

    # Calculate the top-left corner of the ROI based on center position
    x_start = max(0, center_x - roi_width // 2)
    y_start = max(0, center_y - roi_height // 2)

    # Crop and return the ROI from the image
    roi = frame[y_start:y_start + roi_height, x_start:x_start + roi_width]
    return roi


def capture_skin_based_roi(frame):
    # Generate the skin mask
    skin_mask = detect_faces_with_skin_mask(frame)

    # Find contours in the skin mask
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If contours are found, proceed to find the largest skin region
    if contours:
        # Find the largest contour by area, which is likely the main skin region
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Define the ROI based on the bounding box of the largest contour
        roi = frame[y:y + h, x:x + w]

        return roi, (x, y, w, h)  # Return the ROI and bounding box coordinates
    else:
        print("No significant skin-colored region detected.")
        return None, None  # Return None if no skin region is found


def main():
    while True:
        # Prompt user to choose between image or video upload or exit
        choices = ["1. Image Upload", "2. Video Upload", "3. Exit"]
        user_choice = eg.choicebox("Choose an option:", "Upload Choice", choices)

        if user_choice == "1. Image Upload":
            # Handle image upload
            file_path = image_file_explore()
            if not file_path:
                continue  # Return to the main menu

            # Load and process the image
            frame = cv2.imread(file_path)
            if frame is None:
                eg.msgbox("Failed to load the image file.", "Error")
                continue  # Return to the main menu

            # Perform Edge Detection
            edges = detect_faces_with_edge_detection(frame)
            cv2.imshow('Edge Detection', edges)

            # Perform Skin Mask Detection
            mask = detect_faces_with_skin_mask(frame)
            cv2.imshow('Skin Color Mask', mask)

            # Perform Skin Region Detection
            skin = detect_faces_with_skin_region(frame)
            cv2.imshow('Skin Color Region', skin)

            # Display the Fixed ROI around the center of the frame
            face_roi = capture_fixed_roi(frame)
            cv2.imshow("Fixed Face ROI", face_roi)

            # Display dynamically calculated Skin-based ROI
            skin_roi, bound_box = capture_skin_based_roi(frame)
            if skin_roi is not None:
                cv2.imshow("Skin-based ROI", skin_roi)

                # Draw bounding box on the original image
                x, y, w, h = bound_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Original Image with ROI", frame)

            # Wait for user input to close windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif user_choice == "2. Video Upload":
            # Handle video upload
            file_path = video_file_explore()
            if not file_path:
                continue  # Return to the main menu

            # Open the video file
            cap = cv2.VideoCapture(file_path)

            # Retrieve the frame rate of the video to set the playback speed
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 33  # Default to ~30 fps if FPS info is unavailable

            while True:  # Outer loop for continuous replay
                # Reset video to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while cap.isOpened():
                    # Read each frame from the video
                    ret, frame = cap.read()
                    if not ret:
                        print("Reached end of video.")
                        break

                    # Perform Edge Detection
                    edges = detect_faces_with_edge_detection(frame)
                    cv2.imshow('Edge Detection', edges)

                    # Perform Skin Mask Detection
                    mask = detect_faces_with_skin_mask(frame)
                    cv2.imshow('Skin Color Mask', mask)

                    # Perform Skin Region Detection
                    skin = detect_faces_with_skin_region(frame)
                    cv2.imshow('Skin Color Region', skin)

                    # Display the Fixed ROI around the center of each frame
                    face_roi = capture_fixed_roi(frame)
                    cv2.imshow("Fixed Face ROI", face_roi)

                    # Display dynamically calculated Skin-based ROI
                    skin_roi, bound_box = capture_skin_based_roi(frame)
                    if skin_roi is not None:
                        cv2.imshow("Skin-based ROI", skin_roi)
                        # Draw bounding box on the original frame
                        x, y, w, h = bound_box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Original Image with ROI", frame)

                    # Exit the display loop on pressing 'q'
                    if cv2.waitKey(delay) & 0xFF in [ord('q'), ord('Q')]:
                        break  # Exit the video playback loop

                # Ask the user if they want to replay the video
                replay_choice = eg.ynbox("Do you want to replay the video?", "Replay", ["Yes", "No"])
                if not replay_choice:
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Exit the outer replay loop

        elif user_choice == "3. Exit":
            # Exit the program
            print("Exiting the program.")
            break

        else:
            eg.msgbox("Please make a valid selection.", "Error")
            continue  # Continue prompting if no valid choice is selected

    # Close any remaining windows and release resources
    cv2.destroyAllWindows()

main()
