import cv2
import os
from keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("drowsiness_model3.h5")

face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
leye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

#model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        
        r_eye = cv2.resize(r_eye,(64,64))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(64,64)
        r_eye = np.expand_dims(r_eye, axis=-1)  # Add a channel dimension
        r_eye = np.repeat(r_eye, 3, axis=-1)     # Repeat along the last dimension to have 3 channels
        r_eye = np.expand_dims(r_eye, axis=0)  
        
      
        r_eye = r_eye.astype('float32')
        rpred = model.predict(r_eye)    
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY) 
        
        l_eye = cv2.resize(l_eye,(64,64))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(64,64)
        l_eye = np.expand_dims(l_eye, axis=-1)  # Add a channel dimension
        l_eye = np.repeat(l_eye, 3, axis=-1)     # Repeat along the last dimension to have 3 channels
        l_eye = np.expand_dims(l_eye, axis=0)  
        
       
        l_eye = l_eye.astype('float32')
        lpred = model.predict(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]>0.5and lpred[0]>0.5):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        
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


'''
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
model = load_model("drowsiness_model_with_augmentation.h5")

# Open the webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for face
        face_roi = frame[y:y + h, x:x + w]

        # Convert the face ROI to grayscale for left eye detection
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect left eyes in the face ROI
        left_eyes = left_eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=1)

        # Iterate over detected left eyes
        for (lex, ley, lew, leh) in left_eyes:
            # Extract the region of interest (ROI) for left eye
            left_eye = face_roi[ley:ley + leh, lex:lex + lew]

            # Preprocess the left eye image
            left_eye = cv2.resize(left_eye, (64, 64))
            left_eye = left_eye / 255.0
            left_eye = np.reshape(left_eye, (1, 64, 64, 3))

            # Make prediction
            prediction = model.predict(left_eye)

            # Display the prediction result on the frame
            if prediction[0] > 0.3:
                print("Drowsy")
                cv2.putText(frame, "Drowsy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif  0.9:
                print("Not Drowsy")
                cv2.putText(frame, "Not Drowsy", (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw rectangle around the detected left eye
            cv2.rectangle(face_roi, (lex, ley), (lex + lew, ley + leh), (0, 255, 0), 2)

        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
'''