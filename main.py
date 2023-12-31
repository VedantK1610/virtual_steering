import cv2
import mediapipe as mp
import keyboard

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawings_styles = mp.solutions.drawing_styles

previous_right_position = None
d_key_pressed = False
right_threshold = 0.65

previous_left_position = None
a_key_pressed = False
left_threshold = 0.30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)  # Set width to your desired value
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 25)
hands = mp_hands.Hands()

while True:
    frame, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(image)

    if results.multi_hand_landmarks:
        # for hand in results.multi_handedness:
        #     index = hand.classification[0].index

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                image,
                hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            right_hand_x = hand_landmarks.landmark[9].x
            if previous_right_position is not None:
                # Check if the right hand moved to the right
                if right_hand_x > previous_right_position and not d_key_pressed and right_hand_x > right_threshold:
                    keyboard.press('d')
                    d_key_pressed = True
                # Check if the right hand moved back to the center
                elif right_hand_x < previous_right_position and d_key_pressed:
                    keyboard.release('d')
                    d_key_pressed = False
            previous_right_position = right_hand_x

            left_hand_x = hand_landmarks.landmark[9].x
            if previous_left_position is not None:
                # Check if the left hand moved to the left
                if left_hand_x < previous_left_position and not a_key_pressed and left_hand_x < left_threshold:
                    keyboard.press('a')
                    a_key_pressed = True
                # Check if the left hand moved back to the center
                elif left_hand_x > previous_left_position and a_key_pressed:
                    keyboard.release('a')
                    a_key_pressed = False
            previous_left_position = left_hand_x

            right_thumb = hand_landmarks.landmark[4]
            if right_thumb.y < 0.55:  # Adjust the threshold as needed
                keyboard.press_and_release('space')

    cv2.imshow('Virtual Steering', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
