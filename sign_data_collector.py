import cv2
import mediapipe as mp
import numpy as np
import os
import platform
import sys
import time

# === CONFIGURATION ===
SIGN_NAME = "I Love You"
SAMPLES = 500
DATA_DIR = "isl_dataset"
IMAGE_PATH = f"{DATA_DIR}/{SIGN_NAME}"


os.makedirs(IMAGE_PATH, exist_ok=True)

def beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 500)
    elif platform.system() == 'Darwin':
        os.system('say "beep"')
    else:
        print('\a')  # Linux terminal beep

# === Get Next Index from Existing Files ===
def find_next_index(image_path):
    files = [f for f in os.listdir(image_path) if f.endswith('.npy')]
    indices = [int(os.path.splitext(f)[0]) for f in files if f.split('.')[0].isdigit()]
    return max(indices) + 1 if indices else 0

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Camera Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

collected = find_next_index(IMAGE_PATH)
print(f"Starting from sample #{collected}")

while cap.isOpened() and collected < SAMPLES:
    start_time = time.time()

    # Countdown (optional live preview)
    while time.time() - start_time < 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, "Get ready...", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Sign Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    beep()

    # === Capture Frame and Process Landmarks ===
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        print("No hands detected! Skipping sample...")
        continue

    landmark_data = []
    for hand_landmarks in result.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        single_hand = []
        for lm in hand_landmarks.landmark:
            single_hand.extend([lm.x, lm.y, lm.z])
        landmark_data.append(single_hand)

    # Pad missing second hand
    while len(landmark_data) < 2:
        landmark_data.append([0.0] * 63)

    # Concatenate both hands into one flat array
    full_landmarks = np.array(landmark_data[0] + landmark_data[1])

    filename = f"{IMAGE_PATH}/{collected:03d}.npy"
    if not os.path.exists(filename):
        np.save(filename, full_landmarks)
        print(f"Saved: {filename}")
        collected += 1
    else:
        print(f"Skipping {filename}, already exists.")

    cv2.putText(frame, f'{SIGN_NAME}: {collected}/{SAMPLES}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Sign Data Collector", frame)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
