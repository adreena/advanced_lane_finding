import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    undist = cv2.undistort(img,mtx, dist, None,mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret,corners  = cv2.findChessboardCorners(gray,(nx,ny),None)
    im_size= (img.shape[1], img.shape[0])
    if ret == True:
        cv2.drawChessboardCorners(img, (nx,ny),corners,ret)
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        offset = 100
        dst = np.float32([[offset,offset],[im_size[0]-offset,offset],[im_size[0]-offset,im_size[1]-offset],[offset,im_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(undist, M, im_size)
        return warped, M
    #delete the next two lines
    M = None
    warped = np.copy(img) 
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
