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

# Create dataset folder if it does not exist
if not os.path.exists("../data"):
    os.makedirs("../data")

# Open CSV file to store gesture data
file = open("../data/gestures.csv", "a", newline="")
writer = csv.writer(file)

# Ask user for gesture label
label = input("Enter gesture label (A/B/C etc): ")

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save gesture data")
print("Press 'q' to quit")

while True:

    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    result = hands.process(rgb)

    landmark_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks on screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmark coordinates
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

    # Show camera window
    cv2.imshow("Hand Detection", frame)

    # Read keyboard input (ONLY ONCE)
    key = cv2.waitKey(1) & 0xFF

    # Save gesture when 's' is pressed
    if key == ord('s') and landmark_list:
        landmark_list.append(label)
        writer.writerow(landmark_list)
        print("Gesture Saved")

    # Quit program when 'q' is pressed
    if key == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
file.close()