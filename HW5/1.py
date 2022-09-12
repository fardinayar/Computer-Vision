import cv2 as cv
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter = cv.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20, 20),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
old_gray = cv.GaussianBlur(old_gray, ksize=(15, 15), sigmaX=2, sigmaY=2)
p1 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
endpoints1 = []
endpoints0 = []
while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, ksize=(15, 15), sigmaX=3, sigmaY=3)
    # calculate optical flow
    p1, state, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p1, None, **lk_params)
    old_gray = frame_gray.copy()
    endpoints2 = []
    if len(p1) > 0:
        for i,s in enumerate(state):
            if s == 1:
                endpoints2.append(p1[i])
            else:
                endpoints2.append(None)
    if len(endpoints2) > 0 and len(endpoints1) > 0:
        for old,new in zip(endpoints1,endpoints2):
            if old is not None and new is not None:
                old = old[0].astype(int)
                new = new[0].astype(int)
                frame = cv.arrowedLine(frame, old, new, (255,0,0), 2)
    if len(endpoints1) > 0 and len(endpoints0) > 0:
        for old, new in zip(endpoints0, endpoints1):
            if old is not None and new is not None:
                old = old[0].astype(int)
                new = new[0].astype(int)
                frame = cv.arrowedLine(frame, old, new, (0, 255, 0), 2)
    endpoints0 = endpoints1.copy()
    endpoints1 = endpoints2.copy()
    cv.imshow('result', frame)
    videoWriter.write(frame)
    if cv.waitKey(10) == ord('q'):
        break
cap.release()
videoWriter.release()
cv.destroyAllWindows()
