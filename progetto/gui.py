
import PySimpleGUI as sg
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector
import math

def temperatura():
    return 23

def face_detection(frame):
#faccia trovata, frame con contorno della faccia, faccia, distanza
    return False,frame, frame, 12 

def mask_detection(frame):
    return False


sg.theme('LightGreen8')      #layout
layout = [[sg.Image(filename='', key='webcam')]]
window = sg.Window("GUI - progetto FieraElettronica",resizable = False, grab_anywhere = True, margins = (0, 0), icon='images/logoponti.ico').Layout(layout)

cap = cv2.VideoCapture(0)
    
while True:   #event loop
    event, values = window.read(timeout=10)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
    window['webcam'].update(data=imgbytes)

window.close() 