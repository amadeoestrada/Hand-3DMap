#!/usr/bin/env python

"""
    This Python program finds the fundamental matrix using two images of the same scenario.
    A check board pattern and a stereo setup are used. The match coordinates of the corners
    of the check board pattern are found using HarrisCorners and CornersSubPix corrections.

    Then the program uses OpenCV tools to find the Fundamental Matrix F. It uses the tools
    findFundamentalMat and computeCorrespondEpilines.

    Finally, the program draws the epipolar lines and the match coordinates in the original
    images.
"""
__author__ = "Amadeo Estrada"
__date__ = "24 / Jun / 2020"

import numpy as np
import cv2
import glob
import sys
from matplotlib import pyplot as plt

# ---------------------- PARAMETERS SET
nRows = 9
nCols = 6
dimension = 25  # - mm

# Define the calibration folder
workingFolder = "./photos"
imageType = 'png'

# Change the resolution according to the images used
image_res_x = 1920  # input image horizontal resolution
image_res_y = 1080  # input image vertical resolution

# images lists
img = []
photo = []
# ------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRows * nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_r = []  # 2d points in image plane. RIGHT
imgpoints_l = []  # 2d points in image plane. LEFT


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Process the hand image files
filename = workingFolder + "/*." + imageType
images = glob.glob(filename)

# For verification purposes, print file name, if found.
print(len(images))

# Load 2 images
i = 0;
if len(images) > 2:
    print("More than two images found. ABORT!")
    sys.exit()
else:
    for fname in images:
        # -- Read the file and convert in greyscale
        img.append(cv2.imread(fname))
        print("Reading image ", fname)
        i += 1

for index in img:
    #Apply gaussianBlur
    index = cv2.GaussianBlur(index,(3,3),0,0)

    # Apply mask to get the blue component
    hsv = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)

    # Set the values of the color - Hue / Saturation / Value
    lower_blue = np.array([108, 45, 60])        # Upper BLUE value
    upper_blue = np.array([125, 190, 225])      # Lower BLUE value
    
    # Define the first mask approximation
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Creating kernel for morphological transformations
    kernel = np.ones((9, 9), np.uint8)
    
    # Eliminate isolated patterns using Morph Open
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,100)
    
    # Close mask gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,100)

    # Close even further
    mask = cv2.dilate(mask, kernel,100) 

    # Close all the rest of small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,100)
    #img[0] = cv2.morphologyEx(img[0], cv2.MORPH_OPEN, kernel,100)

    # Apply masl to the original imaage
    photo.append(cv2.bitwise_and(index, index, mask=mask))

# Show the results
cv2.imshow('mask1', photo[0])
cv2.imshow('mask2', photo[1])

cv2.imwrite(workingFolder + "/results/photo1.png", photo[0])
cv2.imwrite(workingFolder + "/results/photo2.png", photo[1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

coord1 = np.array([[1152, 407], [1055, 534], [1171, 518], [960, 273],[1007,552],
                   [1016,472],[1059,394],[1126,341],[1149,602],[1245,526],[1091,458],
                   [898,811],[579,386],[533,122],[451,109],[442,507],[597,41],[591,223],
                   [140,54],[1089,276]])
coord2 = np.array([[928, 534], [882, 633], [970, 628], [754, 443], [853,642],[836,582],
                   [848,525],[899,480],[980,695],[1026,634],[893,572],[1153,857],[1774,500],
                   [1729,225],[1355,274],[1350,618],[1783,128],[1786,321],[912,310],[859,436]])

#coord_h1 = np.array([[1152, 407], [1055, 534], [1171, 518], [960, 273]])
#coord_h2 = np.array([[928, 534], [882, 633], [970, 628], [754, 443]])
coord_h1 = np.array([[1152, 407]])
coord_h2 = np.array([[928, 534]])

# Use epipoles to find the matching features
F, mask = cv2.findFundamentalMat(coord1, coord2, cv2.FM_8POINT)

# Extract the camera matrix K from the parameters file:
#filename = workingFolder + "/fund_mat.txt"
#F = np.loadtxt(filename, dtype='float', delimiter=',')

# Function to draw epipolar lines and match circles to images.
def drawlines(img_a, img_b, lines, pts1, pts2):
    """ img_a - image on which we draw the epilines for the points in img_b lines - corresponding epilines
        The lines argument contains an equation for each epipolar line. The for cicle below takes the iterables
        lines, pts1, and pts2 into a tuple assigned to r[0], r[1], r[2], pt1, pt2, for each for iteration.
        Therefore, each iteration will draw a line from the extreme right (x = 0) to the extreme left (x = c)
        and the correspondent coordinates of the matching points on the two images.
    """
    # Assign row, column and color information
    r, c, color_info = img_a.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        # Use random color for each epipolar line
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Set the start of line to the extreme left of the image
        x0, y0 = map(int, [0, -r[2] / r[1]])
        # Set the end of the line to the extreme right of the image
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        # Draw the line using the coordinates above
        img_a = cv2.line(img_a, (x0, y0), (x1, y1), color, 1)
        # Draw the matching coordinates for each image
        img_a = cv2.circle(img_a, tuple(pt1), 5, color, -1)
        img_b = cv2.circle(img_b, tuple(pt2), 5, color, -1)
    return img_a, img_b

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(coord_h1.reshape(-1, 1, 2), 1, F)
lines1 = lines1.reshape(-1, 3)

# noinspection PyUnboundLocalVariable
img5, img6 = drawlines(img[1], img[0], lines1, coord_h2, coord_h1)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(coord_h2.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img[0], img[1], lines2, coord_h1, coord_h2)

# Show plot of the two epipolar line images
#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
#plt.show()

# Press 'q' to exit each image window
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save results in results folder
cv2.imwrite(workingFolder + "/results/img1.png", img5)
cv2.imwrite(workingFolder + "/results/img2.png", img3)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




# - - - - - - - - - START of the programming part that finds F using OpenCV tools - - - - - - - - - - - - -

# Extract the coordinates from the image points list (3 dimension array)
right_coord = np.array(imgpoints_r[0])  # convert from list to array
right_coord = np.squeeze(right_coord)  # Convert to 2 dimension array
# Vertically flip array. Checker board pattern results are ordered from-bottom right to top-left
coord_right = np.flipud(right_coord)

# Extract the coordinates from the image points list (3 dimension array)
left_coord = np.array(imgpoints_l[0])  # convert from list to array
left_coord = np.squeeze(left_coord)  # Convert to 2 dimension array
# Vertically flip array. Checker board pattern results are ordered from-bottom right to top-left
coord_left = np.flipud(left_coord)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find the fundamental matrix and use OpenCV routine for comparison
F1, mask = cv2.findFundamentalMat(right_coord, left_coord, cv2.FM_8POINT)

# Select only inlier points
coord_right = coord_right[mask.ravel() == 1]
coord_left = coord_left[mask.ravel() == 1]

# Function to draw epipolar lines and match circles to images.
def drawlines(img_a, img_b, lines, pts1, pts2):
    """ img_a - image on which we draw the epilines for the points in img_b lines - corresponding epilines
        The lines argument contains an equation for each epipolar line. The for cicle below takes the iterables
        lines, pts1, and pts2 into a tuple assigned to r[0], r[1], r[2], pt1, pt2, for each for iteration.
        Therefore, each iteration will draw a line from the extreme right (x = 0) to the extreme left (x = c)
        and the correspondent coordinates of the matching points on the two images.
    """
    # Assign row, column and color information
    r, c, color_info = img_a.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        # Use random color for each epipolar line
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Set the start of line to the extreme left of the image
        x0, y0 = map(int, [0, -r[2] / r[1]])
        # Set the end of the line to the extreme right of the image
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        # Draw the line using the coordinates above
        img_a = cv2.line(img_a, (x0, y0), (x1, y1), color, 1)
        # Draw the matching coordinates for each image
        img_a = cv2.circle(img_a, tuple(pt1), 5, color, -1)
        img_b = cv2.circle(img_b, tuple(pt2), 5, color, -1)
    return img_a, img_b

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(coord_left.reshape(-1, 1, 2), 2, F1)
lines1 = lines1.reshape(-1, 3)

# noinspection PyUnboundLocalVariable
img5, img6 = drawlines(img1, img2, lines1, coord_right, coord_left)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(coord_right.reshape(-1, 1, 2), 1, F1)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, coord_left, coord_right)

# Show plot of the two epipolar line images
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

# Save results in results folder
cv2.imwrite(workingFolder + "/results/img1.png", img1)
cv2.imwrite(workingFolder + "/results/img2.png", img2)


# - - - - - - - - - END of the programming part that finds F using OpenCV tools - - - - - - - - - - - - -
