
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

#Gesture recognize model
model = load_model('mp_hand_gesture')

#Initialize mediapipe
mp_hands=mp.solutions.hands
hand=mp_hands.Hands(max_num_hands=1,min_tracking_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

classes=[]
with open("gesture.names",'r') as f:
    classes=f.read().split("\n")
print(classes)



while True:
     ret, frame = cap.read()
     frame=cv2.flip(frame,1)
     x,y,c=frame.shape
     imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

     #Get handmark prediction
     result=hand.process(imgrgb)
     print(result)
     #Get landmark detection
     class_name=""
     if result.multi_hand_landmarks:
         landmarks=[]
         for handlms in result.multi_hand_landmarks:
             for lm in handlms.landmark:
                 lmx=int(lm.x*x)
                 lmy=int(lm.y*y)
                 landmarks.append([lmx,lmy])
             #Draw landmark on frame
             mp_draw.draw_landmarks(frame,handlms,mp_hands.HAND_CONNECTIONS)
             #Predict gesture
             prediction=model.predict([landmarks])
             classid=np.argmax(prediction)
             class_name=classes[classid]
     #Show prediction of frame
     cv2.putText(frame,class_name,(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)





     cv2.imshow("img", frame)
     key = cv2.waitKey(1)
     if key == 27:
         break

cap.release()
cv2.waitKey()