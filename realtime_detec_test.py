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
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    def classify(image=frame):
        # 顯示圖片
        cv2.imshow('frame', frame)
        # Use the Inception model to classify the image.
        pred = model.classify(image=frame)
        # Print the scores and names for the top-10 predictions.
        model.print_scores(pred=pred, k=3, only_first_name=True)    

    classify(image=frame)

    # 顯示圖片
    cv2.imshow('frame', frame)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()