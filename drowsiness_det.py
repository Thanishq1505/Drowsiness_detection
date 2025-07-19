import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading
import pygame

# Load pre-trained CNN model
model = load_model("drowsiness_model.h5", compile=False)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Global variables
alarm_on = False
score = 0

# Function to play alarm sound
def alarm():
    global alarm_on
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    alarm_on = False

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (fx, fy, fw, fh) in faces:
        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
        face_roi_color = frame[fy:fy+fh, fx:fx+fw]

        left_eyes = left_eye_cascade.detectMultiScale(face_roi_gray)
        right_eyes = right_eye_cascade.detectMultiScale(face_roi_gray)

        eye_closed = 0

        # Check left eye
        for (ex, ey, ew, eh) in left_eyes:
            eye = face_roi_color[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (64, 64))
            eye = eye.astype('float32') / 255.0
            eye = eye.reshape(1, 64, 64, 3)
            pred = model.predict(eye, verbose=0)
            if np.argmax(pred) == 0:  # Closed
                eye_closed += 1
            break  # Process only one eye

        # Check right eye
        for (ex, ey, ew, eh) in right_eyes:
            eye = face_roi_color[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (64, 64))
            eye = eye.astype('float32') / 255.0
            eye = eye.reshape(1, 64, 64, 3)
            pred = model.predict(eye, verbose=0)
            if np.argmax(pred) == 0:  # Closed
                eye_closed += 1
            break

        # Drowsiness score update
        if eye_closed >= 1:
            score += 1
        else:
            score = max(score - 1, 0)

        break  # Only one face needed

    # Drowsiness alert
    if score > 10:
        cv2.putText(frame, "DROWSY!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=alarm).start()
    else:
        cv2.putText(frame, "Awake", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()