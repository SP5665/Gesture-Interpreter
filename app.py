import streamlit as st
import cv2
import torch
import mediapipe as mp
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from model.gesture_model import GestureModel

st.title("Sign Language Interpreter")

# Load model
model = GestureModel()
model.load_state_dict(torch.load("saved_models/gesture_model.pth"))
model.eval()

labels = ["A", "B", "C", "D", "E", "F"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mirror for natural feel
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmark_list = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

            if len(landmark_list) == 42:
                data = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0)

                output = model(data)
                predicted = torch.argmax(output).item()
                gesture = labels[predicted]

                cv2.putText(
                    img,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="gesture",
    video_processor_factory=GestureProcessor
)

# streamlit run app.py to start the application.