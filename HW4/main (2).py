import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os, shutil
import pandas as pd
import numpy as np
import math

############################# part c ##############################

vid = cv2.VideoCapture(0)
sift = cv2.SIFT_create()
count = 0
frames = []
color = (0, 255, 0)
thickness = 2
while (True):
    ret, frame = vid.read()
    frames.append(frame)
    count += 1
    if count > 2:
        gray_2 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
        keypoints_2, des_2 = sift.detectAndCompute(gray_2, None)
        keypoints_1, des_1 = sift.detectAndCompute(gray_1, None)
        keypoints, des = sift.detectAndCompute(gray, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches_2 = bf.match(des_1, des_2)
        matches_1 = bf.match(des, des_1)
        matches_2 = sorted(matches_2, key=lambda x: x.distance)
        matches_1 = sorted(matches_1, key=lambda x: x.distance)
        for match in matches_2[0:5]:
            end_point2 = [int(keypoints_1[match.queryIdx].pt[0]),int(keypoints_1[match.queryIdx].pt[1])]
            start_point2 = [int(keypoints_2[match.trainIdx].pt[0]),int(keypoints_2[match.trainIdx].pt[1])]
            end_point1 = [int(keypoints[match.queryIdx].pt[0]), int(keypoints[match.queryIdx].pt[1])]
            start_point1 = [int(keypoints_1[match.trainIdx].pt[0]), int(keypoints_1[match.trainIdx].pt[1])]
            frames[2] = cv2.arrowedLine(frames[2], start_point1, end_point1, color, thickness)
            frames[2] = cv2.arrowedLine(frames[2], start_point2, end_point2, color, thickness)
        cv2.imshow('new_frame', frames[2])
        cv2.waitKey(300)
        frames = []
        count = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
