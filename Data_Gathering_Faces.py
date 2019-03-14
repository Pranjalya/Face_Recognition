import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_name = raw_input('\n Enter User Name and press Enter, please : ')
print("\n Look at the camera and smile!")

try:
    os.mkdir("tf_files/Faces_Datasets/"+face_name)
except:
    print("Continue. Folder exists.")

# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    img = cv2.flip(img, +1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img , 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)     
        count += 1
        print("Testing entry ", count)
       
        cv2.imshow('image', img)

            # Press 'ESC' for exiting video       
        
        if count >= 500: # Take 500 face sample and stop video
            break

         # Save the captured image into the datasets folder
        cv2.imwrite("tf_files/Faces_Datasets/" + face_name + '/' + str(count) + ".jpg", img[y:y+h,x:x+w])

        # Image is 126 x 126

        cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        
    if count >= 500: # Take 500 face sample and stop video
        break

    
cam.release()
cv2.destroyAllWindows()
