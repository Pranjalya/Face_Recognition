import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
capture.set(3,640) # set Width
capture.set(4,480) # set Height

print("Press ESC to quit.")

while True:
    ret, img = capture.read()
    img = cv2.flip(img, +1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_color = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
    
    for (x,y,w,h) in faces_color:
        cv2.rectangle(img,(x,y),(x+w,y+h),(100,89,123),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
   # cv2.putText(img, "Kvothe", (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (120,123,156), 2, cv2.LINE_AA)
    cv2.imshow('Testing_', img)
    
    keypress = cv2.waitKey(30)
    if keypress == 27: # press 'ESC' to quit
        break

capture.release()
cv2.destroyAllWindows()