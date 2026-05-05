# Gesture Interpreter <br>

Gesture Interpreter is a real-time hand gesture recognition system that detects hand landmarks using computer vision and predicts gestures using a deep learning model. <br>
The system captures video from the webcam, processes the hand landmarks, and classifies the gesture in real time. <br><br>

## Technologies Used <br>

Python – Programming language used to develop the project. <br>
OpenCV – Used to access the webcam and perform image processing. <br>
MediaPipe – Used to detect hand landmarks and track hand movement. <br>
PyTorch – Used to build and run the gesture classification model. <br>
NumPy – Used for numerical computations and handling arrays. <br><br>

## Project Workflow <br>

1. The webcam captures frames using OpenCV. <br>
2. MediaPipe processes the frame and detects hand landmarks. <br>
3. The landmark coordinates are extracted and converted into numerical features. <br>
4. These features are passed into the trained PyTorch model. <br>
5. The model predicts the gesture label. <br>
6. The predicted gesture is displayed on the screen in real time. <br><br>

## Future Improvements <br>

• Add support for more gestures. <br>
• Improve the training dataset for better accuracy. <br>
• Convert gestures into text or speech output. <br>
• Deploy the project as a web or mobile application. <br><br>

## Running file

- clone this repository
- install the required dependencies and libraries
- in the terminal, run 'streamlit run app.py'

## Collaborators <br>

Srajal Sahu <br>
Srishti Pandey <br>