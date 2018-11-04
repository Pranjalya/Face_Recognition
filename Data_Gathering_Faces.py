import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Enter User Id and press Enter, please : ')
print("\n Look at the camera and smile!")

# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    img = cv2.flip(img, +1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("Faces_Datasets/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', gray)

    k = cv2.waitKey(100) # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 50 face sample and stop video
         break

cam.release()
cv2.destroyAllWindows()