import cv2
from tracker2 import *
import numpy as np

#Creater Tracker Object
tracker = EuclideanDistTracker()

#cap = cv2.VideoCapture("Resources/traffic3.mp4")
cap = cv2.VideoCapture("traffic4.mp4")
# f = 25
# w = int(1000/(f-1))
# print(w)


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
#100,5

#KERNALS Tạo nên các phân tử cho mặt nạ mask
# kernalOp = np.ones((3,3),np.uint8)
# kernalOp2 = np.ones((5,5),np.uint8)
# kernalCl = np.ones((11,11),np.uint8)
# fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height,width,_ = frame.shape
    #print(height,width)
    #540,960


    #Extract ROI cần quan tâm một khung hình cụ thể thay vi cả khung hình 
    roi = frame[50:540,200:960] 

    # #MASKING METHOD 1
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    #DIFFERENT MASKING METHOD 2 -> This is used
    # fgmask = fgbg.apply(roi)
    # ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    # mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    # mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    # e_img = cv2.erode(mask2, kernal_e)


    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])
    #DISPLAY
    cv2.imshow("Mask",mask)
    # cv2.imshow("Mask2",mask2)
    # cv2.imshow("Erode", e_img)
    cv2.imshow("frame",frame)
    # cv2.imshow("ROI", roi)

    key = cv2.waitKey(30)
    if key == 27:  
       break
cap.release()
cv2.destroyAllWindows