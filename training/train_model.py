import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import os

#adds the project root folder to Python’s module search path, so Python can find the model folder.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the neural network model
from model.gesture_model import GestureModel

data = pd.read_csv("data/gestures.csv", header=None) # Load dataset
data = data.sample(frac=1).reset_index(drop=True) # Shuffle dataset to ensure random order during training.
# Split features and labels
X = data.iloc[:, :-1].values   # all columns except last (landmarks)
y = data.iloc[:, -1].values   # last column (gesture label)

#iloc selects data by position.
#X contains the information the model uses to make a prediction.
#y contains the correct answer for each input.

# Convert labels (A,B,C...) → numbers (0,1,2...)
label_map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5}
y = y.map(label_map)

# Convert NumPy arrays → PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = GestureModel() # Create model instance

# Loss function (measures prediction error)
criterion = nn.CrossEntropyLoss()

# Optimizer (updates weights to reduce loss)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200): # Train for 200 epochs
    outputs = model(X) #akes input data and produces predictions.
    loss = criterion(outputs, y) #Compare predicted gestures with true labels.
    optimizer.zero_grad() # Reset gradients
    loss.backward() #Calculates how each weight contributed to the error.
    optimizer.step() #The optimizer updates.

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "saved_models/gesture_model.pth")

print("Model training complete and saved.")