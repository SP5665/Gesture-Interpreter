import cv2
import mediapipe as mp
import csv

# Load MediaPipe hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create hand detection object
hands = mp_hands.Hands(
    static_image_mode=False,      # Use video mode
    max_num_hands=1,              # Detect only one hand
    min_detection_confidence=0.7  # Detection confidence
)

# Open CSV file where gesture data will be stored
file = open("../data/gestures.csv", "a", newline="")
writer = csv.writer(file)

# Ask user for the gesture label (example: A, B, C)
label = input("Enter gesture label: ")

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save gesture")
print("Press 'q' to quit")

while True:

    # Capture frame from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB (required for MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb)

    landmark_list = []

    # If a hand is detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand skeleton on the screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract x and y coordinates of each landmark
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

    # Show webcam window
    cv2.imshow("Hand Detection", frame)

    # Read keyboard input
    key = cv2.waitKey(1)

    # If 's' is pressed → save gesture
    if key == ord('s') and landmark_list:
        landmark_list.append(label)   # add label at end
        writer.writerow(landmark_list)
        print("Gesture Saved!")

    # If 'q' is pressed → exit program
    if key == ord('q'):
        print("Program Closed")
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()

# Close CSV file
file.close()