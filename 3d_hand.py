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
import math
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

# Save the filtered results in a folder
cv2.imwrite(workingFolder + "/results/photo_filter1.png", photo[0])
cv2.imwrite(workingFolder + "/results/photo_filter2.png", photo[1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the coordinates for the calibration points

coord1 = np.array([[1152, 407], [1055, 534], [1171, 518], [960, 273],[1007,552],
                   [1016,472],[1059,394],[1126,341],[1149,602],[1245,526],[1091,458],
                   [898,811],[579,386],[533,122],[451,109],[442,507],[597,41],[591,223],
                   [140,54],[1089,276]])
coord2 = np.array([[928, 534], [882, 633], [970, 628], [754, 443], [853,642],[836,582],
                   [848,525],[899,480],[980,695],[1026,634],[893,572],[1153,857],[1774,500],
                   [1729,225],[1355,274],[1350,618],[1783,128],[1786,321],[912,310],[859,436]])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the coordinates for the palm and fingers
coord_p1 = np.array([[1148, 603],[1061,539],[1155,410], [1244, 526],
                        [1289, 414], [1312, 350], [1330, 279],  # Coordinates thumb
                        [1126, 340], [1096, 288], [1060,213], [998, 143],  # Coordinates index
                        [887, 222], [959, 273], [1009, 331], [1059, 392],  # Coordinates middle
                        [832, 391], [892, 418], [961, 449], [1016, 472],  # Coordinates ring
                        [840, 590], [900, 578], [954, 564], [1007, 552]])  # Coordinates  pinkie
coord_p2 = np.array([[980, 695], [885,636],[931, 535], [1026, 633],
                        [1037, 533], [1012, 477], [999, 415],  # Coordinates thumb
                        [899, 480], [860, 436], [825, 398], [785, 353],  # Coordinates index
                        [856, 523], [805, 482], [756, 443], [717, 414],  # Coordinates middle
                        [836, 582], [787, 565], [727, 543], [695, 527],  # Coordinates ring
                        [852, 640], [811, 647], [711, 653], [734, 656]])  # Coordinates pinkie

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Find F
F, mask = cv2.findFundamentalMat(coord1, coord2, cv2.FM_8POINT)

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
lines1 = cv2.computeCorrespondEpilines(coord_p1.reshape(-1, 1, 2), 1, F)
lines1 = lines1.reshape(-1, 3)

# noinspection PyUnboundLocalVariable
img5, img6 = drawlines(photo[1], photo[0], lines1, coord_p2, coord_p1)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(coord_p2.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(photo[0], photo[1], lines2, coord_p1, coord_p2)

# Save the epipole lines and dots in the results folder
cv2.imwrite(workingFolder + "/results/photo_epipole1.png", img5)
cv2.imwrite(workingFolder + "/results/photo_epipole2.png", img3)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Extract the camera matrix K from the parameters file, for each camera:
filename = workingFolder + "/cameraMatrixL.txt"
K_l = np.loadtxt(filename, dtype='float', delimiter=',')
filename = workingFolder + "/cameraMatrixR.txt"
K_r = np.loadtxt(filename, dtype='float', delimiter=',')

"""
    The intrinsic parameters matrix is:
            [   fx     0    cx  ]      where    alpha = fx
    K =     [    0    fy    cy  ]               beta = fy
            [    0     0     1  ]               cx = u0, cy = v0
    
    The t vector that points from the left camera to the right camera can 
    be found with the following equations.
    
    Left camera is camera 1. Right camera is camera 2.
    
    F = H = [h1_vec, h2_vec, h3_vec] = s · K · [r1_vec, r2_vec, t_vec]
    
    where: F is the fundamental matrix.
            H is a homography
            h*_vec is a vector 3 x 1
            s is the scale factor
            K is the intrinsic parameters matrix
            r*_vec is the rotation vector for camera 1 and camera 2
            t_vec points from camera 1 to camera 2
    then
            t_vec =  lambda · M^-1 · h3_vec         where: lambda = 1/s
            
            but the scale factor is one. Then:
            
                        
            t_vec = M^-1 · [ F13, F23, F33 ]  <- is a vector of 1 x 3 
            
            The result vector 3 x 1 will need to be flipped vertically. 
"""

# Translation vector from computation
K_inv = np.linalg.inv(K_l)
h3 = np.array([[F[0][2], F[1][2], F[2][2]]])
t_vec = K_inv.dot(h3.T)
t_vec = np.flipud(t_vec)


# create 3d point arrays
x_points = np.zeros((coord_p1.shape[0]), dtype=np.float32)
y_points = np.zeros((coord_p1.shape[0]), dtype=np.float32)
z_points = np.zeros((coord_p1.shape[0]), dtype=np.float32)
i = 0

for pt1, pt2 in zip(coord_p1, coord_p2):
    l1 = np.ones((3, 1), dtype=np.float32)
    l1[0] = (pt1[0] - K_l[0][2]) / K_l[0][0]
    l1[1] = (pt1[1] - K_l[1][2]) / K_l[1][1]

    # direction vector of point 2 in image 2
    l2 = np.ones((3, 1), dtype=np.float32)
    l2[0] = (pt2[0] - K_r[0][2]) / K_r[0][0]
    l2[1] = (pt2[1] - K_r[1][2]) / K_r[1][1]

    # calculate theta
    theta1 = math.atan(122 / 533)
    theta2 = math.atan(225 / 1729)

    # Compute the magnitude of L1
    L1_mag = (np.linalg.norm(t_vec)) * math.sin(theta1) / math.sin(theta1 - theta2)

    # compute 3D reconstruction
    L1 = np.ones((3, 1), dtype=np.float32)
    L1 = L1_mag * l1 / np.linalg.norm(l1)

    # Add to 3d_points list
    x_points[i] = L1[0][0]
    y_points[i] = L1[1][0]
    z_points[i] = L1[2][0]
    i += 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot the results

ax = plt.axes(projection='3d')
# draw palm line 1
zline = np.linspace(z_points[0], z_points[1], num=100)
yline = np.linspace(y_points[0], y_points[1], num=100)
xline = np.linspace(x_points[0], x_points[1], num=100)
ax.plot3D(xline, yline, zline, 'gray')
# draw palm line 2
zline = np.linspace(z_points[1], z_points[2], num=100)
yline = np.linspace(y_points[1], y_points[2], num=100)
xline = np.linspace(x_points[1], x_points[2], num=100)
ax.plot3D(xline, yline, zline, 'gray')
# draw palm line 3
zline = np.linspace(z_points[2], z_points[3], num=100)
yline = np.linspace(y_points[2], y_points[3], num=100)
xline = np.linspace(x_points[2], x_points[3], num=100)
ax.plot3D(xline, yline, zline, 'gray')
# draw palm line 2
zline = np.linspace(z_points[3], z_points[0], num=100)
yline = np.linspace(y_points[3], y_points[0], num=100)
xline = np.linspace(x_points[3], x_points[0], num=100)
ax.plot3D(xline, yline, zline, 'red')

# draw thumb line 1
zline = np.linspace(z_points[4], z_points[5], num=100)
yline = np.linspace(y_points[4], y_points[5], num=100)
xline = np.linspace(x_points[4], x_points[5], num=100)
ax.plot3D(xline, yline, zline, 'red')
# draw thumb line 2
zline = np.linspace(z_points[5], z_points[6], num=100)
yline = np.linspace(y_points[5], y_points[6], num=100)
xline = np.linspace(x_points[5], x_points[6], num=100)
ax.plot3D(xline, yline, zline, 'red')

# draw index line 1
zline = np.linspace(z_points[7], z_points[8], num=100)
yline = np.linspace(y_points[7], y_points[8], num=100)
xline = np.linspace(x_points[7], x_points[8], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw index line 2
zline = np.linspace(z_points[8], z_points[9], num=100)
yline = np.linspace(y_points[8], y_points[9], num=100)
xline = np.linspace(x_points[8], x_points[9], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw index line 3
zline = np.linspace(z_points[9], z_points[10], num=100)
yline = np.linspace(y_points[9], y_points[10], num=100)
xline = np.linspace(x_points[9], x_points[10], num=100)
ax.plot3D(xline, yline, zline, 'blue')

# draw middle line 1
zline = np.linspace(z_points[11], z_points[12], num=100)
yline = np.linspace(y_points[11], y_points[12], num=100)
xline = np.linspace(x_points[11], x_points[12], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw middle line 2
zline = np.linspace(z_points[12], z_points[13], num=100)
yline = np.linspace(y_points[12], y_points[13], num=100)
xline = np.linspace(x_points[12], x_points[13], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw middle line 3
zline = np.linspace(z_points[13], z_points[14], num=100)
yline = np.linspace(y_points[13], y_points[14], num=100)
xline = np.linspace(x_points[13], x_points[14], num=100)
ax.plot3D(xline, yline, zline, 'blue')

# draw ring line 1
zline = np.linspace(z_points[15], z_points[16], num=100)
yline = np.linspace(y_points[15], y_points[16], num=100)
xline = np.linspace(x_points[15], x_points[16], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw ring line 2
zline = np.linspace(z_points[16], z_points[17], num=100)
yline = np.linspace(y_points[16], y_points[17], num=100)
xline = np.linspace(x_points[16], x_points[17], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw ring line 3
zline = np.linspace(z_points[17], z_points[18], num=100)
yline = np.linspace(y_points[17], y_points[18], num=100)
xline = np.linspace(x_points[17], x_points[18], num=100)
ax.plot3D(xline, yline, zline, 'blue')

# draw pinkie line 1
zline = np.linspace(z_points[19], z_points[20], num=100)
yline = np.linspace(y_points[19], y_points[20], num=100)
xline = np.linspace(x_points[19], x_points[20], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw pinkie line 2
zline = np.linspace(z_points[20], z_points[21], num=100)
yline = np.linspace(y_points[20], y_points[21], num=100)
xline = np.linspace(x_points[20], x_points[21], num=100)
ax.plot3D(xline, yline, zline, 'blue')
# draw pinkie line 3
zline = np.linspace(z_points[21], z_points[22], num=100)
yline = np.linspace(y_points[21], y_points[22], num=100)
xline = np.linspace(x_points[21], x_points[22], num=100)
ax.plot3D(xline, yline, zline, 'blue')

# -- Set the Position Relative to Camera 3D Plot
plt.suptitle('Position is Relative to Camera')
# -- Display the origin as a pink cross
ax.scatter3D(x_points, y_points, z_points, c='b', marker='P')
plt.show()

# Press 'q' to exit each image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# - - - - - - - - - - - - - - END of the program - - - - - -  - - - - - - - - - - - -
