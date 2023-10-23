import cv2
import numpy as np
import mediapipe as mp
import keyboard

mp_hands=mp.solutions.hands

mp_drawings=mp.solutions.drawing_utils
mp_drawings_styles=mp.solutions.drawing_styles


cap= cv2.VideoCapture(0)
hands=mp_hands.Hands()

while True:
    frame,image = cap.read()
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                image,
                hand_landmarks,mp_hands.HAND_CONNECTIONS
            )

        right_thumb = hand_landmarks.landmark[4]
        if right_thumb.y < 0.5:  # Adjust the threshold as needed
            keyboard.press_and_release('space')

    cv2.imshow('Virtual Steering', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()