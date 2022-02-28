import cv2
import mediapipe as mp
import cvzone
import time
from cvzone.FaceMeshModule import FaceMeshDetector
 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)
detector = FaceMeshDetector(maxFaces=1)

    
while True:
    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) / w
        print(d)
        cvzone.putTextRect(img, f'Distanza: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
        
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        cv2.destroyAllWindows()
        break
        
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    

