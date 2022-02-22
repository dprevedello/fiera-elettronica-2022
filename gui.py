
import PySimpleGUI as sg
import cv2
import numpy as np

def temperatura():
    return 23

def face_detection(frame):
#faccia trovata, frame con contorno della faccia, faccia, distanza
    return false,frame, frame, 12

def mask_detection(frame):
    return false
        

sg.theme('DarkAmber')   
layout = [[sg.Image(filename='', key='webcam')]]

window = sg.Window('interfaccia grafica - aquisizione faccia',
                       layout, location=(800, 400))
                       
cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
    window['webcam'].update(data=imgbytes)
 
window.close()