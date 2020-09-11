import cv2
# 載入分類器

#patha = 'C:\\Users\\RedHat\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier('C:\\Users\\RedHat\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# 從視訊盡頭擷取影片

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1028)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 使用現有影片
#cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # 轉成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 偵測臉部
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # 繪製人臉部份的方框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 顯示成果
    #正常視窗大小
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)  
    #秀出圖片
    cv2.imshow('img', img)                    
    #   Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()