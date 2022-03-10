
import PySimpleGUI as sg
import cv2
import numpy as np
import cvzone
import math
import mediapipe as mp
import time
from cvzone.FaceMeshModule import FaceMeshDetector
from keras.models import load_model

model=load_model("./model2-008.model")

#haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rect_size = 4

#requisiti mask_detection
results={0:'senza mascherina',1:'mascherina ok'}


detector = FaceMeshDetector(maxFaces=1)

def temperatura():
    return 23

def face_detection(frame):
    img, faces = detector.findFaceMesh(frame, draw=False)
    if faces:
        
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 20
        f = 230
        d = (W * f) / w
        d=d+20
        x =max(min(np.array(faces)[0,:,0]),0)
        y = min(max(np.array(faces)[0,:,1]),img.shape[0])
        xn = min(max(np.array(faces)[0,:,0]),img.shape[1])
        yn = max(min(np.array(faces)[0,:,1]),0)
        cvzone.putTextRect(img, f'Distanza: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
        face = img.copy()[yn:y, x:xn]
        img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn], )
                    
        print("AAAAAAAAAAAAAAAAAAAAAAAA")
        #cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return True, img, face, d
    #faccia trovata, frame con contorno della faccia, faccia, distanza
    
    
    return False, frame, frame, None 

def mask_detection(face):
    
    rerect_sized=cv2.resize(face,(224,224))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1,224,224,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    return result


sg.theme('LightGreen8')      #layout
layout = [[sg.Image(filename='', key='webcam')]]
window = sg.Window("GUI - progetto FieraElettronica",resizable = False, grab_anywhere = True, margins = (0, 0), icon='images/logoponti.ico').Layout(layout)

cap = cv2.VideoCapture(0)
cap.set(3, 1680)
cap.set(4, 1120)
pTime = 0
    
while True:   #event loop
    event, values = window.read(timeout=10)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    detected, frame, face, distanza=face_detection(frame)
    mask = mask_detection(face)    
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
    window['webcam'].update(data=imgbytes)
       

window.close() 