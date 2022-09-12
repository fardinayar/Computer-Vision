import cv2
import matplotlib.pyplot as plt

edge = cv2.imread('edge.webp')
c = cv2.Canny(edge,100,250)
plt.imshow(c, cmap='gray')
plt.show()

edge = cv2.GaussianBlur(edge, (9,9), 1.5)
c = cv2.Canny(edge,100,250)
plt.imshow(c, cmap='gray')
plt.show()
