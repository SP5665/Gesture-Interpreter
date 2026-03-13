import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Create dataset folder if not exists
if not os.path.exists("../data"):
    os.makedirs("../data")

file = open("../data/gestures.csv", "a", newline="")
writer = csv.writer(file)

# Ask user for gesture label
label = input("Enter gesture label (A/B/C etc): ")

cap = cv2.VideoCapture(0)

print("Press 's' to save gesture data")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            key = cv2.waitKey(1)

            if key == ord('s'):
                landmark_list.append(label)
                writer.writerow(landmark_list)
                print("Gesture Saved")

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
file.close()