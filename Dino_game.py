import cv2      # to capture the image from the webcam
import pyautogui              # to give inputs to comp
import mediapipe as mp        # models to recognize hand gestures
import time


cap = cv2.VideoCapture(0)         # Using camera index 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

mp_drawing = mp.solutions.drawing_utils

#loop to analyse frames and use mediapipe to recognize hand gestures.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #converting each frame to rgb
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #processing frame with mediapipe
    results = hands.process(image_rgb)

    #checking if there was a gesture by check multi hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            cv2.putText(frame, f"Thumb tip y: {round(thumb_tip.y, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Index tip y: {round(index_finger_tip.y, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Middle tip y: {round(middle_finger_tip.y, 2)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ring tip y: {round(ring_finger_tip.y, 2)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pinky tip y: {round(pinky_tip.y, 2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            is_hand_closed = (
                index_finger_tip.y > thumb_tip.y and
                middle_finger_tip.y > thumb_tip.y and
                ring_finger_tip.y > thumb_tip.y and
                pinky_tip.y > thumb_tip.y
            )

            if is_hand_closed:
                pass
            else:
                pyautogui.press('space')
                time.sleep(0.1)

    #show the frames on the computer
    cv2.imshow('CV', frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
