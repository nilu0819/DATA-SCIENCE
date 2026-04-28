import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands (FIXED)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Create canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables
prev_x, prev_y = 0, 0

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows fix

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Get landmarks
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            distance = euclidean_distance((ix, iy), (tx, ty))

            # Dynamic threshold (better than fixed 40)
            if distance < 0.05 * w:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                # Draw line
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 0, 0), 5)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and frame
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    # Instructions
    cv2.putText(frame, "Pinch (Thumb + Index) to Draw", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'C' to Clear | 'Q' to Quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Virtual Painter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()