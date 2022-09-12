import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.transform import resize



vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    frame = cv2.resize(frame,(200,200))
    frame = frame[50:150, 75:150]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blur, 100, 200)
    result = hough_ellipse(edges, threshold=20,accuracy=20,min_size=40, max_size=60)
    if len(result) > 0:
        result.sort(order='accumulator')
        num = min(len(result),5)
        for n in range(1,num):
            best = list(result[-n])
            yc, xc, a, b = [int(round(x)) for x in best[1:5]]
            orientation = best[5]
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            cx[cx>=75] = 0
            cy[cy>=100] = 0
            frame[cy, cx] = (0, 0, 255)
    frame = cv2.resize(frame, (700, 700))
    edges = cv2.resize(edges, (700, 700))
    cv2.imshow('new_frame', frame)
    cv2.imshow('frame', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()