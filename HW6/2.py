import cv2 as cv
from matplotlib import pyplot as plt

# read left and right images
right_gray = cv.imread('im6.png',0)
left_gray = cv.imread('im2.png',0)


# creates StereoBm object
stereo = cv.StereoBM_create(blockSize=19)
stereo.setNumDisparities(64)
stereo.setTextureThreshold(0)
stereo.setUniquenessRatio(0)
stereo.setSpeckleWindowSize(0)

# computes disparity
disparity = stereo.compute(left_gray, right_gray)

# displays image as grayscale and plotted
disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
plt.imshow(disparity, cmap='gray')
plt.show()
