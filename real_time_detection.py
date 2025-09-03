import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
from collections import deque
import time
import os

# === Load Model and Labels ===
model = load_model('sign_model.h5')
labels = np.load('labels.npy')

# === Initialize Text-to-Speech Engine ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate if necessary

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Initialize Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Prediction Smoothing ===
prediction_history = deque(maxlen=15)
last_spoken = ''
speak_delay = 2  # seconds
last_speak_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    landmark_data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            single_hand = []
            for lm in hand_landmarks.landmark:
                single_hand.extend([lm.x, lm.y, lm.z])
            landmark_data.append(single_hand)

        # Ensure two hands data
        while len(landmark_data) < 2:
            landmark_data.append([0.0] * 63)

        # Prepare input for prediction
        full_landmarks = np.array(landmark_data[0] + landmark_data[1])
        full_landmarks = full_landmarks.reshape(1, -1)

        # Predict
        prediction = model.predict(full_landmarks)
        predicted_label = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Update prediction history
        prediction_history.append(predicted_label)
        most_common = max(set(prediction_history), key=prediction_history.count)

        # Display prediction if stable
        if prediction_history.count(most_common) > 10 and confidence > 0.8:
            cv2.putText(frame, f'{most_common} ({confidence*100:.2f}%)',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Speak the prediction if not recently spoken
            if most_common != last_spoken and (time.time() - last_speak_time) > speak_delay:
                engine.say(most_common)
                engine.runAndWait()
                last_spoken = most_common
                last_speak_time = time.time()
        else:
            cv2.putText(frame, 'Detecting...', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(frame, 'No hands detected', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        prediction_history.clear()
        last_spoken = ''

    cv2.imshow("ISL Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
