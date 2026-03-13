import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model.gesture_model import GestureModel

# load dataset
data = pd.read_csv("data/gestures.csv")

X = data.iloc[:, :-1].values #: → all rows, :-1 → all columns except the last one
y = data.iloc[:, -1].values #selecting the last column of your CSV file.

#X contains the information the model uses to make a prediction.
#y contains the correct answer for each input.

# convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# create model
model = GestureModel()

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
for epoch in range(100):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# save model
torch.save(model.state_dict(), "saved_models/gesture_model.pth")