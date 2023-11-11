import cv2
import numpy as np
import os
import math
import rtde_control
import rtde_io
import time
import threading

Last_time = 999999999999999
convayor = 1
# Initialize the camera capture
cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

# cap = cv2.VideoCapture(0)

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
image_captured = True
capture_counter = 0
speed = 3
acceleration = 3
height = 365
step = 10  # Initial step size

# Initialize the image counter
x = 1
new_point_list = []


def convert(input_list):
    if len(input_list) != 6:
        return "Input list must contain exactly 6 elements."

    part1 = [rx / 1000 for rx in input_list[:3]]
    part2 = [rx * (math.pi / 180) for rx in input_list[3:]]
    output_list = part1 + part2
    return output_list


rtde_c = rtde_control.RTDEControlInterface("192.168.20.35")
rtdeio = rtde_io.RTDEIOInterface("192.168.20.35")
rtdeio.setStandardDigitalOut(3, convayor)

# Define initial position and step size

rtdeio.setStandardDigitalOut(3, 1)

while True:
    Current_time = time.time()

    pos = convert([100, 165, height, 90, 160, 0])
    # time.sleep(2)
    rtde_c.moveJ_IK(pos, speed, acceleration)
    # rtdeio.setStandardDigitalOut(3,1)
    # Read a frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

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
    pink_detected = len(contours)
    if pink_detected > 0:
        pink_detected = True

    if pink_detected:
        # Calculate the centroid of the pink object
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print(cX, cY)

            # Check if the pink object is within the specified rectangle
            if 295 < cX < 345 and 0 < cY < 480:
                Last_time = time.time() + 2
                rtdeio.setStandardDigitalOut(3, 0)
                print("conveyer stop")
            else:
                image_captured = True

            if Current_time > Last_time:
                if image_captured:

                    # time.sleep(2)
                    image_filename = os.path.join(output_folder, f"image_{x}.png")
                    cv2.imwrite(image_filename, frame)
                    print(f"Pink object in the rectangle captured and saved: {image_filename}")

                    #############################################################################

                    fileName = image_filename

                    # Read the image file
                    img = cv2.imread(fileName)
                    # cv2.imshow('Image with 4 corners', img)
                    # img = cv2.flip(img,0)
                    img = cv2.flip(img, 1)
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                    # Convert the image to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow('gray', gray)

                    kernel = np.ones((5, 5), np.uint8)

                    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow('kernel', gray)

                    # Apply a bilateral filter.
                    # This filter smooths the image, reduces noise, while preserving the edges
                    bi = cv2.bilateralFilter(gray, 5, 75, 75)

                    dst = cv2.cornerHarris(bi, 2, 3, 0.02)
                    # cv2.imshow('rectangle', dst)

                    # Dilate the result to mark the corners
                    dst = cv2.dilate(dst, None)
                    # cv2.imshow('rectangle', dst)
                    # Create a mask to identify corners
                    mask = np.zeros_like(gray)

                    # All pixels above a certain threshold are converted to white
                    mask[dst < 0.01 * dst.min()] = 255
                    # cv2.imshow('rectangle', mask)

                    # Convert corners from white to red.
                    # img[dst > 0.01 * dst.max()] = [0, 0, 255]

                    # Create an array that lists all the pixels that are corners
                    coordinates = np.argwhere(mask)

                    # Convert array of arrays to lists of lists
                    coordinates_list = [l.tolist() for l in list(coordinates)]

                    # Convert list to tuples
                    coordinates_tuples = [tuple(l) for l in coordinates_list]

                    # Create a distance threshold
                    thresh = 30


                    # Compute the distance from each corner to every other corner.
                    def distance(pt1, pt2):
                        (x1, y1), (x2, y2) = pt1, pt2
                        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        return dist


                    # Keep corners that satisfy the distance threshold
                    coordinates_tuples_copy = coordinates_tuples
                    i = 1
                    for pt1 in coordinates_tuples:
                        for pt2 in coordinates_tuples[i::1]:
                            if (distance(pt1, pt2) < thresh):
                                coordinates_tuples_copy.remove(pt2)
                        i += 1

                    # Place the corners on a copy of the original image
                    img2 = img.copy()
                    # img2 = cv2.flip(img2,0)
                    # img2 = cv2.flip(img2,1)
                    list_mate = []
                    for pt in coordinates_tuples:
                        pos = tuple(reversed(pt))
                        # if ((0 < pos[0] < 640) and (110 < pos[1] < 440)):
                        # print(tuple(reversed(pt)))  # Print corners to the screen

                        list_mate.append(tuple(reversed(pt)))
                        # img = cv2.flip(img,0)
                        # img = cv2.flip(img,1)

                        cv2.circle(img2, tuple(reversed(pt)), 10, (0, 0, 255), -1)
                    # print(list_mate)

                    points = np.array(list_mate)

                    # หาจุดศูนย์กลางของข้อมูล
                    center = np.mean(points, axis=0)

                    # คำนวณมุมของแต่ละจุดที่สุดท้าย
                    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

                    # เรียงลำดับจุดโดยใช้มุมเพื่อให้มันหมุนตามเข็มนาฬิกา
                    sorted_points = points[np.argsort(angles)]

                    sorted_pairs = [(int(x), int(y)) for x, y in sorted_points]
                    print(sorted_pairs)

                    cv2.imshow('Image with 4 corners', img2)
                    cv2.imwrite('harris_corners_jeans.jpg', img2)
                    cv2.rectangle(img2, (150, 75), (500, 400), (0, 0, 255), 2)


                    #############################################################################################################

                    def map_ValueX(px_XMin, px_XMax, arm_XMin, arm_XMax, pixel_X):
                        robotX = ((pixel_X - px_XMin) * (arm_XMax - arm_XMin) / (px_XMax - px_XMin)) + arm_XMin
                        return robotX


                    def map_ValueY(px_YMin, px_YMax, arm_YMin, arm_YMax, pixel_Y):
                        robotY = ((pixel_Y - px_YMin) * (arm_YMax - arm_YMin) / (px_YMax - px_YMin)) + arm_YMin
                        return robotY


                    def separate_coordinates(point_list):
                        x_values = []
                        y_values = []

                        for point in point_list:
                            if len(point) == 2:
                                x, y = point
                                xn = map_ValueX(0, 640, 188, 348, x)
                                yn = map_ValueY(0, 480, -10, 108, y)
                                x_values.append(xn)
                                y_values.append(yn)

                        return x_values, y_values


                    def create_points_from_coordinates(x_values, y_values):
                        points = []

                        for x, y in zip(x_values, y_values):
                            points.append((x, y))

                        return points


                    x_values, y_values = separate_coordinates(sorted_pairs)
                    new_point_list = create_points_from_coordinates(x_values, y_values)

                    for i in new_point_list:
                        mapx = i[0]
                        mapy = i[1]
                        pos = convert([mapx, mapy, height, 90, 160, 0])
                        rtde_c.moveJ_IK(pos, speed, acceleration)
                        # time.sleep(1)
                    pos = convert([100, 165, height, 90, 160, 0])
                    print(pos)
                    x += 1
                    image_captured = False
                    Last_time = 999999999999999
                    rtdeio.setStandardDigitalOut(3, 1)

        pink_detected = 0

        # Display the frame with pink object detection
    cv2.rectangle(frame, (295, 0), (345, 480), (255, 0, 0), 2)

    # cv2.rectangle(frame, (0, 110), (640, 440), (0, 255, 0), 2)
    cv2.imshow("Pink Object Detection", frame)
    # Extract x and y values into separate lists
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()