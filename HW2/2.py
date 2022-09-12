import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

cap = cv2.VideoCapture('istockphoto-1179819982-640_adpp_is.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

times = []
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(gray, (5,5))
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 200 , param1=130, param2=10, minRadius=30,
                               maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    out.write(frame)
    times.append(time.time() - start_time)

print(f'time for each frame {np.mean(np.array(times))}')
cap.release()
out.release()
