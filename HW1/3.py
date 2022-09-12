import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('illumination.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


indoor = cv2.imread('illumination.png')[:,:300]
indoor[indoor[:,:,0]==255] = 0
indoor = cv2.cvtColor(indoor, cv2.COLOR_BGR2RGB)
indoor = cv2.cvtColor(indoor,cv2.COLOR_RGB2HSV)
indoor[:,:,2] = cv2.equalizeHist(indoor[:,:,2])
indoor = cv2.cvtColor(indoor,cv2.COLOR_HSV2RGB)
plt.subplot(1,2,1)
plt.imshow(indoor, cmap="hsv")

indoor = cv2.imread('illumination.png')[:,300:]
indoor[indoor[:,:,0]==255] = 0
indoor = cv2.cvtColor(indoor, cv2.COLOR_BGR2RGB)
indoor = cv2.cvtColor(indoor,cv2.COLOR_RGB2HSV)
indoor[:,:,2] = cv2.equalizeHist(indoor[:,:,2])
indoor = cv2.cvtColor(indoor,cv2.COLOR_HSV2RGB)
plt.subplot(1,2,2)
plt.imshow(indoor, cmap="hsv")

plt.show()
