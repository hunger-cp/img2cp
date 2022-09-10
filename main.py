import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from kmeans import *
import houghbundler
from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse



# Import CP
IMAGE_PATH = "images/cp.png"  #@param {type:"string"}
img = cv2.imread(IMAGE_PATH)

# Isolating Points if you need them
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

# Binary image (Needed for Hough Lines)
# adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# thresh_type = cv2.THRESH_BINARY_INV
# bin_img = cv2.adaptiveThreshold(gray, 255, adapt_type, thresh_type, 11, 2)
_, bin_img = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

# Cropping image to square
box = np.where(gray==0)
crop = gray[box[0][0]:box[0][-1],box[1][0]:box[1][-1]]
imgcrop = img[box[0][0]:box[0][-1],box[1][0]:box[1][-1]]
# Isolating lines if you need them
# Find the edges in the image using canny detector
rho, theta, t =1, np.pi/8, 500 # CPs dont have a lot of noise so we can use a low rho and threshold
"""
lines = cv2.HoughLinesP(crop, rho=rho, theta=theta, threshold=t, minLineLength = 20, maxLineGap = 10)
print(len(lines))
"""
#print("bundling")
#bundler = houghbundler.HoughBundler(min_distance=5, min_angle=11.25)
#lines = bundler.process_lines(lines)
"""
coords = corner_peaks(corner_harris(crop), min_distance=5, threshold_rel=0.05)
coords_subpix = corner_subpix(crop, coords, window_size=1)

fig, ax = plt.subplots()
ax.imshow(imgcrop, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis()
plt.show()
# Displaying lines
"""
"""
dst = cv2.cornerHarris(np.float32(crop),2,1,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(crop,np.float32(centroids),(5,5),(-1,-1),criteria)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(imgcrop,(int(x),int(y)),5,(36,255,12),-1)
plt.imshow(imgcrop)
plt.show()
"""
corners = cv2.goodFeaturesToTrack(crop, 0, 0.1, 5)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(imgcrop,(int(x),int(y)),5,(36,255,12),-1)
plt.imshow(imgcrop)
plt.show()

"""
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(crop,(x1,y1),(x2,y2),(0),2)
"""
"""print(len(lines))
segmented = segment_by_angle_kmeans(lines)
print(len(segmented[1]))

print(len(segmented[0]))
for x in range(0, len(segmented[0])):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(imgcrop, (x1,y1),(x2,y2),(0,255,0),2)
for x in range(0, len(segmented[1])):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(imgcrop, (x1,y1),(x2,y2),(0,0,0),2)
"""
# For HoughLines Diplay
"""
a,b,c = lines.shape
for i in range(a):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
    cv2.line(imgcrop, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
"""
"""
def get_contours(img, crop):
    contours, hierarchies = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, peri * 0.04, True)
            cv2.drawContours(img, approx, -1, (0, 0, 255), 8)
get_contours(imgcrop, crop)
"""