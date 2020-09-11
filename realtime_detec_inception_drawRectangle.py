import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Functions and classes for loading and using the Inception model.
import inception

inception.maybe_download()

model = inception.Inception()

# 選擇第1隻攝影機
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1028)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    # 從攝影機擷取三張影像
    _, img0 = cap.read()
    _, img1 = cap.read()
    ret, frame = cap.read()

    #以下openCV框出前景-----------------------------------------------------------------------------------------
    # 轉成灰階
    gray1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1,(7,7),0)
    blur2 = cv2.GaussianBlur(gray2,(5,5),0)
    d = cv2.absdiff(blur1, blur2)
    ret, th = cv2.threshold( d, 10, 255, cv2.THRESH_BINARY )
    dilated=cv2.dilate(th, None, iterations=1)
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours] 
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    markColor=(0,255,0)
    cv2.drawContours(img0, cnt, -1, markColor, 2)
    cv2.rectangle(img0,(x,y),(x+w,y+h), markColor,2)
    # 顯示成果
    #秀出圖片
    #cv2.imshow('img', img0)                    
    #以上openCV框出前景-----------------------------------------------------------------------------------------

    #以下inceptionV3部分-----------------------------------------------------------------------------------------

    def classify(image=frame):
        # 顯示圖片
        cv2.imshow('frame', frame)
        # Use the Inception model to classify the image.
        pred = model.classify(image=frame)
        # Print the scores and names for the top-10 predictions.
        model.print_scores(pred=pred, k=3, only_first_name=True)    
    classify(image=frame)

    #以上inceptionV3部分-----------------------------------------------------------------------------------------

    # 顯示圖片
    cv2.imshow('frame', img0)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()