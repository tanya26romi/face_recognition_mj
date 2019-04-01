import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


cam = cv2.VideoCapture(0)
font =cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = im[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))


        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,255), 2)

        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #if(conf<50):
        if(Id==1):
            Id="Tanya"
        if(Id==2):
            Id="shirashti"
        if(Id==3):
            Id="Shubham"
        if(Id==4):
            Id="Siddhant"
        if(Id==5):
            Id="Somauli"
        if(Id==6):
            Id="Vijay Sir"

        else:
            Id="Unknown"
        cv2.putText(im, str(Id), (x,y+h), font, fontScale, fontColor)
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
