# 라이브러리 불러오기

import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time



#믹서 초기화
mixer.init()
#사운드 사운드 불러오기
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml') #얼굴인식
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml') #왼쪽눈 인식
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml') #오른쪽눈 인식



lbl=['Close','Open'] #눈 감았는지 판단하기 위한 리스트

model = load_model('./models/cnnCat2.h5') #학습모델 불러오기
path = os.getcwd() #경로가져오기
cap = cv2.VideoCapture(0) #카마레 영상 가져오기
font = cv2.FONT_HERSHEY_COMPLEX_SMALL #  작은 크기의 손글씨 글꼴 사용하기
#초기화 설정하기
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

#반복실행
while(True):
    ret, frame = cap.read() #캡춰한 영상의 프레임 추출
    height,width = frame.shape[:2] #높이와 넓이 사각형 크기 추출

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #그래이색으로 추출한 이미지 변환
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))#얼굴 인식
    left_eye = leye.detectMultiScale(gray) #왼쪽눈 인식
    right_eye =  reye.detectMultiScale(gray) #오른쪽눈 인식

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED ) #사각형 그리기

    for (x,y,w,h) in faces: #얼굴안의 사각형 정보 발견되면 사각형으로 표시
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye: #우측눈 분석
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye) #눈 인식 예측
        if(rpred[0]==1): # 1이면 눈 뜬거임
            lbl='Open' 
        if(rpred[0]==0): # 0이면 눈 감은거임
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye: #왼쪽눈 분석
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):# 1이면 눈 뜬거임
            lbl='Open'   
        if(lpred[0]==0):# 0이면 눈 감은거임
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0): #두 눈을 모두 감았다면,
        score=score+1
        # "Take a break and start studying again."
        cv2.putText(frame,"Take a break and start studying again.",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else: #그렇지 않다면 집중하고 있는 것임
        score=score-1
        # "You're concentrating. You're doing good."
        cv2.putText(frame,"You're doing good.",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #집중을 하고 있는 것 같지 않다면..
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
