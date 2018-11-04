import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'Faces_Datasets'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples = []
    ids= []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

print ("\n Training Datasets.")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into Trainer/trainer.yml
recognizer.save('Trainer/trainer.yml')

print("\n {0} faces trained.".format(len(np.unique(ids))))