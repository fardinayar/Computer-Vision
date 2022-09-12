import cv2
import numpy as np


def draw_flow(img, flow, step, x, y,color):
    h, w = img.shape[:2]
    if len(x) == 0:
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.round(lines).astype(int)
    vis = img
    for (x1, y1), (x2, y2) in lines:
        if x1 - x2 < 2 and y1 - y2 < 2:
            continue
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 2)
    return vis, lines[:, 1, 0], lines[:, 1, 1]


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter = cv2.VideoWriter('output_2.avi', fourcc, 10.0, (640, 480))
_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
x, y = [], []
prev2 = []
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 3, 3)
    vis = frame
    if len(prev2) > 0:
        flow = cv2.calcOpticalFlowFarneback(prev2, prev_gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
        vis, x, y = draw_flow(frame, flow, 100, x, y, (0, 0, 255))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
    vis, x, y = draw_flow(vis, flow, 100, x, y, (0, 255, 0))
    cv2.imshow('Optical flow', vis)
    videoWriter.write(vis)
    prev2 = prev_gray.copy()
    prev_gray = gray.copy()
    if cv2.waitKey(20) == ord('q'):
        break
cap.release()
videoWriter.release()
cv2.destroyAllWindows()