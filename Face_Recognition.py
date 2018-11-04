import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX 

id = 0

names = ['None', 'Pranjalya', 'Abhishek', 'Pallavi', 'Vinay'] 
name = "Unknown"



# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
#minW = 0.1*cam.get(3)
#minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, +1)                 # Flip Horizontally
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for(x,y,w,h) in faces:
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 2)
     #   cv2.putText(gray, "PRANJALYA", (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (120,123,156), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(confidence)
        
#        if (confidence < 100):
#            face_name = names[id]
#            confidence = "  {0}%".format(round(100 - confidence))
  #      else:
   #         face_name = "unknown"
    #        confidence = "  {0}%".format(round(100 - confidence))
        
    #    cv2.putText(img, str(face_name), (x+5,y-5), font, 1, (255,255,255), 2)
     #   cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 
    
        if(confidence<120):
            if(id==1):
                id="Mummyji"
            else:
                id="Unknown"

        cv2.putText(gray, str(id), (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (120,123,156), 2)
        cv2.putText(gray, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,0), 1) 
    


    cv2.imshow('video',gray) 
    k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()