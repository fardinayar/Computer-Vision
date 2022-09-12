# import the opencv library
import cv2
import time

# define a video capture object
import numpy as np
from matplotlib import pyplot as plt

n = 40

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frames = []
i = 0
while(True):
    i += 1
    ret, frame = cap.read()
    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
    if i<n+1:
        continue
    frames.append(frame/255)
    mean = np.mean(np.array(frames[len(frames)-n:]), axis=0)
    cv2.imshow('Mean',mean)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
