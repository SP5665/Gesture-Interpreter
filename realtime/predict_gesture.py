import torch
import cv2
import mediapipe as mp
import sys
import os

# allow importing from project folders
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.gesture_model import GestureModel

# load trained model
model = GestureModel()
model.load_state_dict(torch.load("saved_models/gesture_model.pth"))
model.eval() #tells pytorch we are predicting, not training

# gesture labels
labels = ["A","B","C","D","E","F"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
while True:
    # frame = image captured from webcam.
    # ret = whether frame was captured successfully.
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmark_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

        if len(landmark_list) == 42: # 21 landmarks * 2 (x and y), verify before prediction.
            #PyTorch models work with tensors, not Python lists.
            data = torch.tensor(landmark_list, dtype=torch.float32)

            data = data.unsqueeze(0) #add batch dimension (1,42) since model expects batches of data.
            output = model(data)
            predicted = torch.argmax(output).item() #finds the index of the largest value.
            gesture = labels[predicted] #convert index to gesture label.

            #writes the predicted gesture on screen.
            cv2.putText(
                frame,
                gesture,
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()