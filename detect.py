import cv2
from ultralytics import YOLO  # Make sure you have the YOLO library installed

# Load the YOLO model
model = YOLO("/Users/vinodarava/Downloads/outputsfile/runs/detect/train/weights/best.pt")

# Open the webcam (use 0 for the default camera, or change to the camera index you want)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Display the results on the frame
    annotated_frame = results[0].plot()  # Plot the results on the frame

    # Show the frame with detections
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
