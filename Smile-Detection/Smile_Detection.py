 
import cv2
import numpy
from random import randrange 
 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab webcam feed (if o means default webcam)
webcam = cv2.VideoCapture(0)

#Show the current frame
while True:

    #Read the current frame from the web video stream
    successful_frame_read,frame = webcam.read()

    #if there an error abor
    if not successful_frame_read:
        break

    #chage to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    #Blur all exprect face some time smile scale detect unwanted location to train system

    
    for(x,y,w,h) in faces:

        #Draw a rectangle around the smiles
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)
        
        #Get the sub frame(using numpy N-Dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]
        print("Theface frame is ",the_face)
        
        #chage to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=20)

        #Find all simles in the face
        for(x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face,(x_,y_),(x_ + w_,y_ + h_),(50,50,200),4)

    print(faces)


    #Show the current frame
    cv2.imshow('Smile Detector',frame)

    #Display 0
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
cv2.waitKey()