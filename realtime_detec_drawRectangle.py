import cv2
import numpy as np
# 載入分類器

# 從視訊盡頭擷取影片
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1028)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

t0 = cap.read()[1]
t1 = cap.read()[1]
# 使用現有影片
#cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    #_, img = cap.read()
    _, img0 = cap.read()
    _, img1 = cap.read()
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
    cv2.imshow('img', img0)                    
    #   Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()