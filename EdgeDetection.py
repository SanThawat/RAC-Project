import cv2  # OpenCV library
import numpy as np  # NumPy scientific computing library
import math  # Mathematical functions

# The file name of your image goes here
fileName = 'image_1.png'

# Read the image file
img = cv2.imread(fileName)
cv2.imshow('Image with 4 corners', img)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)



#gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow('Image with 4 corners', gray)

kernel = np.ones((5,5),np.uint8)


gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('kernel', gray)



# Apply a bilateral filter.
# This filter smooths the image, reduces noise, while preserving the edges
bi = cv2.bilateralFilter(gray, 5, 75, 75)


dst = cv2.cornerHarris(bi, 2, 3, 0.02)
#cv2.imshow('rectangle1', dst)

# Dilate the result to mark the corners
dst = cv2.dilate(dst, None)
#cv2.imshow('rectangle2', dst)
# Create a mask to identify corners
mask = np.zeros_like(gray)
cv2.imshow('rectangle3', mask)
# All pixels above a certain threshold are converted to white
mask[dst < 0.01 * dst.min()] = 255
cv2.imshow('rectangle4', mask)

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
list_mate = []
for pt in coordinates_tuples:
    pos = tuple(reversed(pt))
    if((150<pos[0]<500) and (100<pos[1]<420)):
        #print(tuple(reversed(pt)))  # Print corners to the screen
        list_mate.append(tuple(reversed(pt)))
        cv2.circle(img2, tuple(reversed(pt)), 10, (0, 0, 255), -1)
print(list_mate)


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
cv2.rectangle(img, (150, 75), (500, 400), (0, 0, 255), 2)

#cv2.imshow('rectangle', img)

# Exit OpenCV
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()