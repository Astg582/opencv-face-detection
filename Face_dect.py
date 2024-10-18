import cv2
import numpy as np

# Loading the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setting up the camera
cap = cv2.VideoCapture(0)

# Getting the camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# Known average face height (in cm)
KNOWN_HEIGHT = 20.0  # Average face height for adults in centimeters
# Note: For children, this height will be less than 20 cm.

# Calibrated focal length from your measurements (e.g., 650)
FOCAL_LENGTH = 650  # Adjust this based on your calibration results

# Function to calculate the distance to the face
def calculate_distance(focal_length, real_height, face_height_in_pixels):
    if face_height_in_pixels == 0:  # Prevent division by zero
        return float('inf')  # Return infinity if no face is detected
    return (focal_length * real_height) / face_height_in_pixels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Drawing a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate the distance to the face
        distance = calculate_distance(FOCAL_LENGTH, KNOWN_HEIGHT, h)
        #print(f"Estimated distance to face: {distance:.2f} cm")

        # Display the distance on the video frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the exit message at the bottom of the frame
    cv2.putText(frame, "Press 'q' to exit", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video with face detection and distance estimation
    cv2.imshow('Face Detection & Distance Estimation', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

