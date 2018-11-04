import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, +1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(120,120,120),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  

    cv2.imshow('video', gray)

    k = cv2.waitKey(30)
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()