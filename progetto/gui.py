
import PySimpleGUI as sg
import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(200, (5,5), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2,2),

    Conv2D(150, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(50, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.load_weights("./model.h5")
results={0: 'NO mascherina', 1: 'mascherina OK'}


detector = FaceMeshDetector(maxFaces=1)

def temperatura():
    #from smbus2 import SMBus
    #from mlx90614 import MLX90614
    #bus = SMBus(1)
    #sensor = MLX90614(bus, address=0x5A)
    #print "Ambient Temperature :", sensor.get_ambient()
    #print "Object Temperature :", sensor.get_object_1()
    #bus.close()
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
        t = 36.5
        if d > 65:
            img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn],colorC=(0,255,255),colorR=(0,255,255) )
            cvzone.putTextRect(img, f'troppo lontano ({int(d)}cm)',
                            (face[10][0] - 100, face[10][1] - 50), 
                            scale=2)
        elif d>=50 and d<=65:
            
            if t<30:
                img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn],colorC=(0,255,255),colorR=(0,255,255) )
                cvzone.putTextRect(img, f'avvicina la mano al sensore',
                           (face[10][0] - 190, face[10][1] - 50),
                           scale=2)
            else:
                if t >= 37.5:
                    img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn],colorC=(0,0,255),colorR=(0,0,255) )
                    cvzone.putTextRect(img, f'temperatura: {int(t)}',
                            (face[10][0] - 100, face[10][1] - 100),
                            scale=2)
                    cvzone.putTextRect(img, f' troppo alta!',
                        (face[10][0] - 100, face[10][1] + 400),
                        scale=2)
                else:
                    img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn],colorC=(0,255,0),colorR=(0,255,0) )
                    cvzone.putTextRect(img, f'temperatura: {int(t)}',
                            (face[10][0] - 100, face[10][1] - 100),
                            scale=2)
        elif d<50:
            img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn],colorC=(0,255,255),colorR=(0,255,255) )
            cvzone.putTextRect(img, f'troppo vicino ({int(d)}cm)',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
        face = img.copy()[yn:y, x:xn]
        #img = cvzone.cornerRect(img,[x,yn,xn-x,y-yn], )
                    
                                    
        #cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return True, img, face, d
        
    #faccia trovata, frame con contorno della faccia, faccia, distanza
    
    
    return False, frame, frame, None 


def mask_detection(frame):
    face = detector.findFaceMesh(frame, draw=False)
    rerect_sized=cv2.resize(frame,(200,200))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1,200,200,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    idx_max = np.argmax(result, axis=1)[0]
    label = results[idx_max] if result[0][idx_max] > 0.7 else ""
    return  label


sg.theme('LightGreen8')      #layout
layout = [[sg.Image(filename='', key='webcam')]]
window = sg.Window("GUI - progetto FieraElettronica",resizable = False, grab_anywhere = True, margins = (0, 0), icon='images/logoponti.ico').Layout(layout)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
while True:   #event loop
    event, values = window.read(timeout=1)

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame,1)
        detected, frame, face, distanza=face_detection(frame)
        mask = mask_detection(frame)    
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
        window['webcam'].update(data=imgbytes)
        
    
       

window.close() 