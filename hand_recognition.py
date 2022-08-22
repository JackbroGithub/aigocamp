import cv2
import mediapipe as mp
import time
import numpy as np

def get_str_guester(up_fingers,list_lms):
    if len(up_fingers)==2 and up_fingers[0]==16 and up_fingers[1]==12:
        str_guester="C"
    elif len(up_fingers)==1 and up_fingers[0]==12:
        str_guester="G"



cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                list_lms=[]
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    list_lms.append([int(xPos),int(yPos)])

                list_lms=np.array(list_lms,dtype=np.int32)
                hull_index=[0,1,2,3,7,11,15,19,18,20,17]
                hull=cv2.convexHull(list_lms[hull_index,:])

                cv2.polylines(img,[hull],True,(0,0,255),2)

                n_fig=-1
                ll=[16,12]
                up_fingers=[]

                for i in ll:
                    pt=(int(list_lms[i][0]),int(list_lms[i][1]))
                    dist=cv2.pointPolygonTest(hull,pt,True)
                    if dist<0: #dist<0 手指在線外
                        up_fingers.append(i)
                str_guester=get_str_guester(up_fingers,list_lms)
                cv2.putText(img,"和弦是: %s" %str_guester,(90,90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,0),4,cv2.LINE_AA)
                



        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

