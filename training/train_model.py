import torch
import pandas as pd
from model.gesture_model import GestureModel

data = pd.read_csv("data/gestures.csv")

X = data.iloc[:, :-1].values #: → all rows, :-1 → all columns except the last one
y = data.iloc[:, -1].values #selecting the last column of your CSV file.

#X contains the information the model uses to make a prediction.
#y contains the correct answer for each input.

model = GestureModel() #This creates an instance of your neural network.

# training loop here