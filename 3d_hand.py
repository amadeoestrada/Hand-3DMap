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
#cv2.imshow('mask1', photo[0])
#cv2.imshow('mask2', photo[1])

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
coord_h1 = np.array([[1152, 407],[1091,458],[533,122]])
coord_h2 = np.array([[928, 534],[893,572],[1729,225]])

# Use epipoles to find the matching features
F, mask = cv2.findFundamentalMat(coord1, coord2, cv2.FM_8POINT)

# Extract the camera matrix K from the parameters file, for each camera:
filename = workingFolder + "/cameraMatrixL.txt"
K_l = np.loadtxt(filename, dtype='float', delimiter=',')
filename = workingFolder + "/cameraMatrixR.txt"
K_r = np.loadtxt(filename, dtype='float', delimiter=',')
filename = workingFolder + "/cameraDistortionL.txt"
D_l = np.loadtxt(filename, dtype='float', delimiter=',')
filename = workingFolder + "/cameraDistortionR.txt"
D_r = np.loadtxt(filename, dtype='float', delimiter=',')

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

# Save results in results folder
cv2.imwrite(workingFolder + "/results/img1.png", img5)
cv2.imwrite(workingFolder + "/results/img2.png", img3)

cv2.imshow('imgNOTRectified1', img3)
cv2.imshow('imgNOTRectified2', img5)

coord1t = coord1.T
coord2t = coord2.T

E, mask2 = cv2.findEssentialMat(coord1, coord2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)

E1 = np.zeros((3, 3), dtype=np.float32)

E1 = K_r.T
E1 = E1.dot(F)
E1 = K_l.dot(E1)

#R2, t, E, F = cv2.stereoCalibrate()
points, R, t, mask2 = cv2.recoverPose(E1, coord1, coord2)
points, R3, t3, mask2 = cv2.recoverPose(E, coord1, coord2)


R1, R2, P1, P2, Q, alpha1, alpha2 = cv2.stereoRectify(K_l,D_l,K_r,D_r,(1920,1080),R3,t3,flags = cv2.CALIB_ZERO_DISPARITY)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
R5 = np.zeros((3, 3), dtype=np.float32)
R5 = R2

#R3[0][0] = R2[0][2]
#R3[1][0] = R2[1][2]
#R3[2][0] = R2[2][2]
#R2[0][2] = R2[0][0]
#R2[1][2] = R2[1][0]
#R2[2][2] = R2[2][0]
#R2[0][0] = R3[0][0]
#R2[1][0] = R3[1][0]
#R2[2][0] = R3[2][0]

#R2 = R2.T

#R3[0][1] = R2[0][1]
#R2[0][1] = R2[2][1]
#R2[0][1] = R3[0][1]

#R3[1][0] = R2[2][1]
#R2[2][1] = R2[1][0]
#R2[2][1] = R3[1][0]

mapx1, mapy1 = cv2.initUndistortRectifyMap(K_l, D_l, R1, K_l,
                                               #(1920,1080),
                                               img[0].shape[:2],
                                               cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(K_r, D_r, R2, K_r,
                                               #(1920,1080),
                                               img[0].shape[:2],
                                               cv2.CV_32F)



img_rect1 = cv2.remap(img[0], mapx1, mapy1, cv2.INTER_LINEAR)
img_rect2 = cv2.remap(img[1], mapx2, mapy2, cv2.INTER_LINEAR)

#img_rect2 = cv2.flip(img_rect2, 1)

# draw the images side by side
total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
              img_rect1.shape[1] + img_rect2.shape[1], 3)
img_8 = np.zeros(total_size, dtype=np.uint8)
img_8[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img_8[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# draw horizontal lines every 25 px accross the side by side image
for i in range(20, img_8.shape[0], 25):
    cv2.line(img_8, (0, i), (img_8.shape[1], i), (255, 0, 0))

cv2.imshow('imgRectified1', img_rect1)
cv2.imshow('imgRectified2', img_rect2)
cv2.imshow('imgRectified', img_8)

cv2.imwrite(workingFolder + "/results/img_rect1.png", img_rect1)
cv2.imwrite(workingFolder + "/results/img_rect2.png", img_rect2)

#World coordinates in pixels
XYZ = np.array([[0],[0],[0],[0]])
#xyd = np.array([[1091],[458],[198],[1]])
xyd = np.array([[533],[1729],[2000],[1]])
XYZ = Q.dot(xyd)
XYZ[0] = XYZ[0]/XYZ[3]
XYZ[1] = XYZ[1]/XYZ[3]
XYZ[2] = XYZ[2]/XYZ[3]


# Press 'q' to exit each image window
cv2.waitKey(0)
cv2.destroyAllWindows()



# - - - - - - - - - END of the program - - - - - - - - - - - - -
