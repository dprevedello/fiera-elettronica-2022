
import PySimpleGUI as sg
import cv2
import numpy as np
import cvzone
import math
import mediapipe as mp
import time
from cvzone.FaceMeshModule import FaceMeshDetector


detector = FaceMeshDetector(maxFaces=1)

def temperatura():
    return 23

def face_detection(frame):
    img, faces = detector.findFaceMesh(frame, draw=False)
    if faces:
        print (faces)
        print(min(faces[0,:,0]), max(faces[0,:,0]))
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) / w
        d=d-6
        
        cvzone.putTextRect(img, f'Distanza: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
                                    
        #cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return True, img, img, d
        
    #faccia trovata, frame con contorno della faccia, faccia, distanza
    return False, frame, frame, None 

def mask_detection(frame):
    return False


sg.theme('LightGreen8')      #layout
layout = [[sg.Image(filename='', key='webcam')]]
window = sg.Window("GUI - progetto FieraElettronica",resizable = False, grab_anywhere = True, margins = (0, 0), icon='images/logoponti.ico').Layout(layout)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0
    
while True:   #event loop
    event, values = window.read(timeout=10)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    detected, frame, face, distanza=face_detection(frame)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
    window['webcam'].update(data=imgbytes)
       

window.close() 