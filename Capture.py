import cv2
import numpy as np
import os

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Create a folder to save the captured images
output_folder = "captured_images"
os.makedirs(output_folder, exist_ok=True)

# Delete all existing images in the captured_images folder
existing_images = os.listdir(output_folder)
for image_file in existing_images:
    if image_file.endswith(".png"):
        os.remove(os.path.join(output_folder, image_file))

# Initialize variables for pink object detection and capturing
pink_detected = False
image_captured = False
capture_counter = 0

# Initialize the image counter
x = 1

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading from the camera")
        break

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the pink color in HSV
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([180, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)

    # Find contours in the pink color mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if a pink object is detected
    pink_detected = len(contours) > 0

    if pink_detected:
        # Calculate the centroid of the pink object
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX, cY)

            # Check if the pink object is within the specified rectangle
            if 295 < cX < 345 and 215 < cY < 265:
                if not image_captured:
                    image_filename = os.path.join(output_folder, f"image_{x}.png")
                    cv2.imwrite(image_filename, frame)
                    print(f"Pink object in the rectangle captured and saved: {image_filename}")
                    x += 1
                    image_captured = True
            else:
                image_captured = False

    # Display the frame with pink object detection
    cv2.rectangle(frame, (295, 215), (345, 265), (255, 0, 0), 2)
    cv2.imshow("Pink Object Detection", frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()