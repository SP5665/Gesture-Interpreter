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

## Project Folder Structure <br>

Gesture-Interpreter <br>
│ <br>
├── model <br>
│ └── gesture_model.py <br>
│ <br>
├── realtime <br>
│ └── predict_gesture.py <br>
│ <br>
├── saved_models <br>
│ └── gesture_model.pth <br>
│ <br>
├── dataset <br>
│ <br>
└── README.md <br><br>

## Installation Steps <br>

### Step 1 – Clone the repository <br>

Command used: <br>

```
git clone https://github.com/YOUR_USERNAME/Gesture-Interpreter.git
```

Explanation: <br>
The `git clone` command downloads the project from GitHub to your local computer so you can run and modify the code. <br><br>

### Step 2 – Navigate to the project folder <br>

Command used: <br>

```
cd Gesture-Interpreter
```

Explanation: <br>
The `cd` command means "change directory". It moves the terminal into the project folder so that commands can be executed inside the project. <br><br>

### Step 3 – Install required libraries <br>

Command used: <br>

```
pip install torch opencv-python mediapipe numpy
```

Explanation: <br>
The `pip install` command installs the required Python libraries needed to run the project. <br>
Torch installs the deep learning framework, OpenCV allows webcam access, MediaPipe detects hand landmarks, and NumPy helps with numerical data processing. <br><br>

## Running the Project <br>

Command used: <br>

```
python realtime/predict_gesture.py
```

Explanation: <br>
This command runs the main Python script that opens the webcam and starts real-time gesture recognition. <br><br>

When the program runs: <br>

• The webcam will open automatically. <br>
• Your hand will be detected using MediaPipe. <br>
• The trained model will predict the gesture. <br>
• The predicted gesture will appear on the screen. <br><br>

Press **Q** on the keyboard to stop the program and close the webcam window. <br><br>

## Git Commands Used in This Project <br>

### Initialize Git Repository <br>

Command: <br>

```
git init
```

Explanation: <br>
This command initializes a Git repository in the project folder so version control can be used. <br><br>

### Add Files to Git <br>

Command: <br>

```
git add .
```

Explanation: <br>
The `git add .` command adds all files in the project folder to the staging area so they can be committed. <br><br>

### Commit Changes <br>

Command: <br>

```
git commit -m "Initial project commit"
```

Explanation: <br>
A commit saves a snapshot of the current project files in Git with a message describing the changes. <br><br>

### Connect to GitHub Repository <br>

Command: <br>

```
git remote add origin https://github.com/YOUR_USERNAME/Gesture-Interpreter.git
```

Explanation: <br>
This command connects your local Git project to a remote repository on GitHub. <br><br>

### Push Code to GitHub <br>

Command: <br>

```
git push -u origin main
```

Explanation: <br>
The `git push` command uploads your local project files to GitHub so they are stored online and can be shared with others. <br><br>

## Future Improvements <br>

• Add support for more gestures. <br>
• Improve the training dataset for better accuracy. <br>
• Convert gestures into text or speech output. <br>
• Deploy the project as a web or mobile application. <br><br>

## Collaborators <br>

Srajal Sahu <br>
Srishti Pandey <br>