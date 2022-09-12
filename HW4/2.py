# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:32:08 2018
@author: Saurav
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

detector=cv2.xfeatures2d.FREAK_create()
kp_detector=cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
trainImg=cv2.imread("book1.jpg",0)
trainImg= cv2.GaussianBlur(trainImg, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)

trainKP,trainDesc=kp_detector.detectAndCompute(trainImg,None)
trainKP,trainDesc=detector.compute(trainImg,trainKP)

trainImg1=cv2.drawKeypoints(trainImg,trainKP,None,(255,0,0),4)
plt.imshow(trainImg1)
plt.show()

cam=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_f.avi', fourcc, 20.0, (640,480))
end2 = []
end3 = []
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)

    QueryImg= cv2.GaussianBlur(QueryImg, ksize=(15,15), sigmaX=3, sigmaY=3)

    queryKP,queryDesc= kp_detector.detectAndCompute(QueryImg,None)
    queryKP, queryDesc = detector.compute(QueryImg, queryKP)


    matches_o = bf.match(trainDesc, queryDesc)

    print(len(matches_o))
    print(sorted(matches_o, key=lambda x: x.distance)[1].distance)

    trh = 45

    end1 = []
    if(len(matches_o)>0):
        for m in matches_o:
            if m.distance > trh:
                end1.append(None)
                continue
            end1.append(tuple(np.round(queryKP[m.trainIdx].pt).astype(int)))
        print("Match Found")
        if len(end1) != 0 and len(end2) != 0:
            for i in range(len(end1)):
                if end1[i] != None and end2[i] != None:
                    QueryImgBGR = cv2.arrowedLine(QueryImgBGR, end2[i], end1[i], (0, 0, 222), 2)
        if len(end3) != 0 and len(end2) != 0:
            for i in range(len(end2)):
                if end2[i] != None and end3[i] != None:
                    QueryImgBGR = cv2.arrowedLine(QueryImgBGR, end2[i], end3[i], (0, 222, 0), 2)

        end3 = end2.copy()
        end2 = end1
    else:
        pass
    cv2.imshow('result',QueryImgBGR)
    out.write(QueryImgBGR)
    if cv2.waitKey(10)==ord('v'):
        break
cam.release()
out.release()
cv2.destroyAllWindows()