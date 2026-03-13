import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model.gesture_model import GestureModel

# load dataset
data = pd.read_csv("data/gestures.csv")

X = data.iloc[:, :-1].values #: → all rows, :-1 → all columns except the last one
y = data.iloc[:, -1].values #selecting the last column of the CSV file.

#iloc selects data by position.
#X contains the information the model uses to make a prediction.
#y contains the correct answer for each input.

# convert to tensors
# convert NumPy array → PyTorch tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# create model
model = GestureModel()

# loss function measures how wrong the model's prediction is.
criterion = nn.CrossEntropyLoss()

# optimizer updates the neural network weights to reduce loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
for epoch in range(100):

    outputs = model(X) #akes input data and produces predictions.
    loss = criterion(outputs, y) #Compare predicted gestures with true labels.

    optimizer.zero_grad() #sets all gradients to zero.
    loss.backward() #Calculates how each weight contributed to the error.
    optimizer.step() #The optimizer updates.

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}") #Every 10 epochs, the loss is printed.

# save model
torch.save(model.state_dict(), "saved_models/gesture_model.pth")