import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
            thumb_tip = 4

            landmarks = hand_landmarks.landmark

            thumb_is_up = landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x \
                if landmarks[mp_hands.HandLandmark.WRIST].x < landmarks[thumb_tip].x \
                else landmarks[thumb_tip].x > landmarks[thumb_tip - 1].x

            fingers_up = 0
            if thumb_is_up:
                fingers_up += 1
            for tip in finger_tips:
                # If the tip is above the PIP joint (closer to the wrist), count as "up"
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers_up += 1

            cv2.putText(frame, f'Fingers: {fingers_up}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Finger Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()