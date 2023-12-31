

import cv2
import numpy as np
from tensorflow.keras.models import load_model
# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
model = load_model("drowsiness_model3.h5")

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
            if prediction[0] > 0.5:
                cv2.putText(frame, "Drowsy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Not Drowsy", (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw rectangle around the detected left eye
            cv2.rectangle(face_roi, (lex, ley), (lex + lew, ley + leh), (0, 255, 0), 2)

        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()