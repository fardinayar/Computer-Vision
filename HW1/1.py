import cv2
import matplotlib.pyplot as plt

hist1 = cv2.imread('Hist1.webp')
hist1 = cv2.cvtColor(hist1, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(hist1)

hist2 = cv2.imread('Hist2.webp')
hist2 = cv2.cvtColor(hist2, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,2)
plt.imshow(hist2)
plt.show()

gray_hist1 = cv2.cvtColor(hist1, cv2.COLOR_RGB2GRAY)
plt.subplot(1,2,1)
plt.imshow(gray_hist1, cmap='gray')

gray_hist2 = cv2.cvtColor(hist2, cv2.COLOR_RGB2GRAY)
plt.subplot(1,2,2)
plt.imshow(gray_hist2, cmap='gray')
plt.show()