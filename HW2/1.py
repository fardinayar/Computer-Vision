import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('sudoku.jpg')

dst = cv2.Canny(src, 50, 200)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

linesP = cv2.HoughLinesP(dst,1, np.pi / 180, 200, None,0, 50)
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(src, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", src)
cv2.waitKey()
