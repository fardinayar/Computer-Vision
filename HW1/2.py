import cv2
import numpy as np
from matplotlib import pyplot as plt

hist1 = cv2.imread('Hist1.webp')
hist1 = cv2.cvtColor(hist1, cv2.COLOR_BGR2RGB)
gray = hist1[:,:,2]
plt.imshow(gray, cmap='gray')
plt.show()