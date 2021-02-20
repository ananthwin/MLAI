#1)Collect all of face images
#2)images should convert in balck and white(grayscale)
#3)Train the algoritm to read images
#pip install opencv-python #OpenComputerVision(GrayScale) Provied by some people,If you go their github  we hae
#libray and datafiles
#Downlaod the haarcasade_frontalface_default
import cv2
from random import randrange 

# Load some pre-trained data on face frontal from opencv(haar cascade algorithm)
#What is harcascade haar is devloper and casacde all  in images
#Haar features: Edge features, Lne Features and Four-rectangle features, cascading over and over on the images
#Supervised-human labled and Unsupervised :computer itslef learn
#step1:Collect lot data
#step2: We gotta Test Every Haar Feature on every training image,(Type,size,location and each HF gives us a number right or wrong)
#whichever haar feature matches the training images closet is our First winner
#Step3: Collect all feature 1000s is good and that is

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Choose an image to detect faces in
#img = cv2.imread('5.JPG')

#To capture video from webcam(if you put 0 it will go to the auto detect cam)
webcam = cv2.VideoCapture(0)

###Iterate froever over frames
while True:
     #### Rad the current frame
    successful_fram_read, frame = webcam.read()
    
    #Must conver to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect the faces  
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangles around the faces 2 is (thickness of the rectangle,) and (0,255,0) is rgb and others are facecoordinates
    for (x,y,w,h) in face_coordinates:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)     

    #Display the image with faces
    cv2.imshow('Clever Programmer Face Detector',frame) 
    
    #if wait key not there then it wont work
    key = cv2.waitKey(1)

    ### Stop if Q key is pressed(q ascii number)
    if key==81 or key==113:
        break

### Release webcam
webcam.release()