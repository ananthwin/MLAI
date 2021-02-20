#1)Collect all of face images
#2)images should convert in balck and white(grayscale)
#3)Train the algoritm to read images
#pip install opencv-python #OpenComputerVision(GrayScale) Provied by some people,If you go their github  we hae
#libray and datafiles
#Downlaod the haarcasade_frontalface_default
import cv2
from random import randrange 
# Load some pre-trained data on face frontal from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
img = cv2.imread('Give some image path jpg or png ')
#Must conver to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#detect the faces  
face_coordinates =  trained_face_data.detectMultiScale(grayscaled_img)
#Draw rectangles around the faces 2 is (thickness of the rectangle,) and (0,255,0) is rgb and others are facecoordinates
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10) 
#(x,y,w,h) = face_coordinates[0]
#cv2.rectangle(img,(586, 226),(586+648, 226+648),(0,255,0),2)
#print(face_coordinates) 
#Display the image with faces
cv2.imshow('Clever Programmer Face Detector',img) 
cv2.waitKey()
print("Code Completed")