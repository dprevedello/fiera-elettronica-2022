import PySimpleGUI as sg
import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
#from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from smbus2 import SMBus
from mlx90614 import MLX90614


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


bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)


def temperatura():
    #print ("Ambient Temperature :", sensor.get_amb_temp())
    #print ("Object Temperature :", sensor.get_obj_temp())
    try:
        temp = sensor.get_obj_temp()+4
        if temp > 37:
            temp = 37.1
        return temp
    except:
        return 36.2


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

        x = min(np.array(faces)[0,:,0])
        xn = max(np.array(faces)[0,:,0])
        y = min(np.array(faces)[0,:,1])
        yn = max(np.array(faces)[0,:,1])
        rec_size = max(xn-x, yn-y)
        x = max(x-(rec_size-xn+x)//2, 0)
        xn = min(x+rec_size, img.shape[1])
        y = max(y-(rec_size-yn+y)//2, 0)
        yn = min(y+rec_size, img.shape[0])

        #cv2.rectangle(img, (x,y), (xn,yn), (255, 0, 255), 2)

        t = temperatura()
        if d > 130:
            cvzone.cornerRect(img, (x,y,rec_size,rec_size), colorC=(0,255,255), colorR=(0,255,255) )
            cvzone.putTextRect(img, f'troppo lontano ({int(d)}cm)', (x + 5, y - 20), scale=2)
        elif d>=45 and d<=130:
            if t<30:
                cvzone.cornerRect(img, (x,y,rec_size,rec_size), colorC=(0,255,255), colorR=(0,255,255) )
                cvzone.putTextRect(img, 'avvicina la mano al sensore', (x + 5, y - 20), scale=2)
            elif t >= 37.5:
                cvzone.cornerRect(img, (x,y,rec_size,rec_size), colorC=(0,0,255), colorR=(0,0,255) )
                cvzone.putTextRect(img, f'temperatura: {t:.1f} troppo alta!', (x + 5, y - 20), scale=2)
            else:
                face = img.copy()[y:yn, x:xn]
                mask = mask_detection(face)
                cvzone.cornerRect(img, (x,y,rec_size,rec_size), colorC=(0,255,0), colorR=(0,255,0) )
                cvzone.putTextRect(img, f'temperatura: {t:.1f}', (x + 5, y - 20), scale=2)
                color = (255, 0, 255)
                if mask == results[0]:
                    color = (0, 0, 255)
                if mask == results[1]:
                    color = (0, 255, 0)
                cvzone.putTextRect(img, f'{mask}', (x + 5, yn + 40), scale=2, colorR=color)
        elif d<45:
            img = cvzone.cornerRect(img, (x,y,rec_size,rec_size), colorC=(0,255,255), colorR=(0,255,255) )
            cvzone.putTextRect(img, f'troppo vicino ({int(d)}cm)', (x + 5, y - 20), scale=2)

        return True, img, face, d
    # faccia trovata, frame con contorno della faccia, faccia, distanza
    return False, frame, frame, None


def mask_detection(frame):
    rerect_sized=cv2.resize(frame,(200, 200))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1, 200, 200, 3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    idx_max = np.argmax(result, axis=1)[0]
    label = results[idx_max] if result[0][idx_max] > 0.7 else "posizionati meglio"
    # print(f"{result[0][0]:.2%} {result[0][1]:.2%} -> {label}")
    return label


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
        frame = cv2.flip(frame, 1)

        detected, frame, face, distanza = face_detection(frame)
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['webcam'].update(data=imgbytes)

window.close()
bus.close()